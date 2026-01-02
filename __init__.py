import os
import io
import time
import subprocess
import tempfile
import platform
import logging
from urllib.request import urlretrieve, URLError
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union, Any, Dict, List
import numpy as np
import torch
from PIL import Image, ImageOps
from comfy import utils

# Configure logging for better error tracking
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("SaveImageCompressed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PNGQUANT_URLS = {
    "windows": "https://github.com/bmgjet/Ai_Models_Downloads/raw/refs/heads/main/pngquant.exe",
    "linux": "https://github.com/bmgjet/Ai_Models_Downloads/raw/refs/heads/main/pngquant"
}
os.makedirs(BASE_MODELS_DIR, exist_ok=True)


def get_model_dir(*sub: str) -> str:
    """Get or create a model directory with comprehensive error handling."""
    try:
        path = os.path.join(BASE_MODELS_DIR, *sub)
        os.makedirs(path, exist_ok=True)
        return path
    except Exception as e:
        logger.error(f"Failed to create model directory: {e}")
        # Fallback to a safe temporary directory
        fallback_dir = os.path.join(tempfile.gettempdir(), "comfyui_models", *sub)
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


@lru_cache(maxsize=1)
def get_pngquant_path() -> str:
    """Ensure pngquant is downloaded and return its path with full error handling."""
    models_dir = get_model_dir("pngquant")
    system = platform.system().lower()
    exe_name = "pngquant.exe" if system == "windows" else "pngquant"
    url = PNGQUANT_URLS.get(system, PNGQUANT_URLS["linux"])
    pngquant_path = os.path.join(models_dir, exe_name)

    # Ensure the directory exists before checking for the file
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(pngquant_path):
        print("[pngquant] Downloading...")
        try:
            urlretrieve(url, pngquant_path)
            os.chmod(pngquant_path, 0o755)
            print("[pngquant] Ready")
        except (URLError, OSError) as e:
            logger.error(f"Failed to download pngquant: {e}")
            print(f"[pngquant] Download failed, compression will be unavailable: {e}")
            # Return None to indicate unavailability, but don't fail
            return None

    # Verify the file exists and is executable
    if os.path.exists(pngquant_path):
        os.chmod(pngquant_path, 0o755)
        return pngquant_path
    else:
        logger.warning(f"pngquant binary not found at {pngquant_path}")
        return None


def safe_remove_file(filepath: Optional[str]) -> None:
    """Safely remove a file, ignoring errors if file doesn't exist or can't be removed."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not remove file {filepath}: {e}")


def tensor_to_pil(image_tensor: torch.Tensor, node_name: str = "tensor_to_pil") -> Image.Image:
    """Convert a normalized tensor to PIL Image with comprehensive error handling."""
    try:
        # Ensure tensor is on CPU and detached
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image_tensor).__name__}")

        image_tensor = image_tensor.detach().cpu()

        # Ensure tensor is in valid range
        if torch.isnan(image_tensor).any():
            logger.warning(f"[{node_name}] NaN detected in tensor, replacing with zeros")
            image_tensor = torch.nan_to_num(image_tensor, nan=0.0)

        # Clip values to valid range [0, 1]
        image_tensor = torch.clamp(image_tensor, 0, 1)

        # Convert to numpy
        img_np = image_tensor.numpy()

        # Handle different tensor formats
        if img_np.ndim == 4:
            # Remove batch dimension if present
            if img_np.shape[0] == 1:
                img_np = img_np[0]
            else:
                logger.warning(f"[{node_name}] Unexpected batch dimension, using first image")
                img_np = img_np[0]

        if img_np.ndim == 3:
            # Handle (C, H, W) format
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            elif img_np.shape[0] == 4:
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = img_np[:, :, :3]  # Remove alpha channel if present
            elif img_np.shape[0] == 1:
                img_np = img_np[0]

        # Convert to uint8
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        # Determine image mode and create PIL Image
        if img_np.ndim == 2:
            return Image.fromarray(img_np, mode="L")
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            return Image.fromarray(img_np, mode="RGB")
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            return Image.fromarray(img_np, mode="RGBA")
        else:
            # Final fallback: convert to RGB
            logger.warning(f"[{node_name}] Unexpected image format, converting to RGB")
            img = Image.fromarray(img_np)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img

    except Exception as e:
        logger.error(f"[{node_name}] Failed to convert tensor to PIL: {e}")
        # Create a fallback black image
        return Image.new("RGB", (512, 512), (0, 0, 0))


def compress_image(pil_img: Image.Image, quality_min: int = 65, quality_max: int = 80) -> bytes:
    """Compress a PIL image using pngquant with robust fallback to uncompressed PNG."""
    pngquant_path = get_pngquant_path()

    # Fast path: if pngquant is unavailable, skip to uncompressed PNG
    if pngquant_path is None or not os.path.exists(pngquant_path):
        return save_pil_to_bytes(pil_img)

    tmp_in_path: Optional[str] = None
    tmp_out_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            tmp_in_path = tmp_in.name

        # Save image to temp file
        try:
            pil_img.save(tmp_in_path, format="PNG")
        except Exception as e:
            logger.warning(f"Failed to save to temp file, trying byte conversion: {e}")
            safe_remove_file(tmp_in_path)
            return save_pil_to_bytes(pil_img)

        tmp_out_path = tmp_in_path.replace(".png", "_out.png")

        cmd = [
            pngquant_path,
            f"--quality={quality_min}-{quality_max}",
            "--force",
            tmp_in_path,
            "--output",
            tmp_out_path
        ]

        # Run compression with timeout
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=30  # 30 second timeout
            )
        except subprocess.TimeoutExpired:
            logger.warning("pngquant timed out, using uncompressed PNG")
            return save_pil_to_bytes(pil_img)
        except subprocess.CalledProcessError as e:
            logger.warning(f"pngquant process error (code {e.returncode}), using uncompressed PNG: {e.stderr}")
            return save_pil_to_bytes(pil_img)

        # Read compressed output
        if os.path.exists(tmp_out_path):
            try:
                with open(tmp_out_path, "rb") as f_out:
                    compressed_bytes = f_out.read()
                # Verify we got valid data
                if len(compressed_bytes) > 0:
                    return compressed_bytes
                else:
                    logger.warning("pngquant returned empty output, using uncompressed PNG")
            except (OSError, IOError) as e:
                logger.warning(f"Failed to read compressed output: {e}")

        return save_pil_to_bytes(pil_img)

    except Exception as e:
        logger.error(f"Unexpected error in compress_image: {e}")
        return save_pil_to_bytes(pil_img)

    finally:
        # Cleanup temp files
        safe_remove_file(tmp_in_path)
        safe_remove_file(tmp_out_path)


def save_pil_to_bytes(pil_img: Image.Image, format_name: str = "PNG") -> bytes:
    """Save PIL image to bytes as a fallback method."""
    try:
        buf = io.BytesIO()
        # Ensure image is in a compatible format
        if pil_img.mode not in ["RGB", "RGBA", "L"]:
            pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format=format_name)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"Failed to save PIL to bytes: {e}")
        # Ultimate fallback: create a minimal valid PNG
        try:
            fallback_img = Image.new("RGB", (1, 1), (0, 0, 0))
            buf = io.BytesIO()
            fallback_img.save(buf, format="PNG")
            buf.seek(0)
            return buf.read()
        except Exception:
            # This should never happen, but return something if all else fails
            return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'


def create_progress_callback(node_name: str, on_progress: Optional[Any] = None):
    """Create a standardized logging/progress callback with error handling."""

    def log(msg: str, prog: Optional[float] = None):
        try:
            if on_progress:
                on_progress(msg, prog)
            else:
                print(f"[{node_name}] {msg}")
        except Exception as e:
            # Never let the callback fail the node
            print(f"[{node_name}] {msg} (progress callback error: {e})")

    return log


def validate_input(value: Any, name: str, empty_error: str) -> Any:
    """Validate that a required input is not empty with graceful handling."""
    if not value:
        logger.warning(f"Empty input detected for {name}")
        # Return None instead of raising exception to allow fallback behavior
        return None
    return value


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid characters."""
    if not filename:
        return "output.png"

    # Get just the filename, no path
    filename = os.path.basename(filename)

    # Remove any path traversal attempts
    filename = filename.replace("..", "_")

    # Replace invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Ensure it has an extension
    _, ext = os.path.splitext(filename)
    if not ext:
        filename += ".png"

    # Limit filename length
    max_length = 255
    if len(filename) > max_length:
        filename = filename[:max_length - len(ext)] + ext

    return filename


def safe_ensure_directory(path: str) -> bool:
    """Safely ensure a directory exists, returning success status."""
    try:
        os.makedirs(path, exist_ok=True)
        return os.path.isdir(path)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def safe_open_file(path: str, mode: str = 'rb'):
    """Open a file with comprehensive error handling."""
    try:
        return open(path, mode)
    except (FileNotFoundError, OSError, PermissionError) as e:
        logger.error(f"Failed to open file {path}: {e}")
        raise


def get_unique_save_path(output_base: str, filename: str) -> Tuple[str, str]:
    """
    Get a unique save path. If the file already exists, append a number to the filename.
    Handles folder paths like 'foldername\\filename'.

    Returns:
        Tuple of (full_save_path, final_filename)
    """
    # Extract folder and filename from user input
    folder_part = os.path.dirname(filename)
    base_filename = os.path.basename(filename)

    # Sanitize filename
    base_filename = sanitize_filename(base_filename)

    if not base_filename:
        base_filename = "output.png"

    # Get name and extension
    name_without_ext, ext = os.path.splitext(base_filename)
    if not ext:
        ext = ".png"

    # If user specified a folder, create it relative to output_base
    if folder_part:
        folder_part = sanitize_filename(folder_part)
        full_folder = os.path.join(output_base, folder_part)
        safe_ensure_directory(full_folder)
        working_base = full_folder
    else:
        working_base = output_base

    # Build initial path
    final_path = os.path.join(working_base, base_filename)

    # If file exists, add number suffix
    counter = 1
    while os.path.exists(final_path):
        new_filename = f"{name_without_ext}_{counter}{ext}"
        final_path = os.path.join(working_base, new_filename)
        counter += 1
        # Safety limit to prevent infinite loops
        if counter > 1000:
            break

    return final_path, os.path.basename(final_path)


class SaveImageCompressed:
    """Save images to compressed PNG format with comprehensive error handling."""

    skip_validation = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "output.png", "multiline": False}),
                "quality_min": ("INT", {"default": 65, "min": 0, "max": 100}),
                "quality_max": ("INT", {"default": 80, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "pngquant"

    def save_images(self, images, filename, quality_min=65, quality_max=80):
        """Save images with multiple layers of fallback to ensure it never fails."""
        log = create_progress_callback("SaveImageCompressed")

        # Validate inputs with fallbacks
        if not isinstance(images, torch.Tensor):
            log("Invalid images type, attempting conversion")
            if hasattr(images, '__iter__'):
                try:
                    images = torch.stack([torch.as_tensor(img) for img in images])
                except Exception:
                    log("Failed to convert images to tensor")
                    return {"ui": {"error": "Invalid images format"}}
            else:
                return {"ui": {"error": "images must be tensor-like"}}

        # Detach and move to CPU safely
        try:
            images = images.detach().cpu()
        except Exception as e:
            log(f"Failed to detach/move to CPU: {e}")
            return {"ui": {"error": "Failed to process images"}}

        # Create progress bar
        try:
            pbar = utils.ProgressBar(images.shape[0])
        except Exception as e:
            log(f"Failed to create progress bar: {e}")
            pbar = None

        # Get timestamp for fallback naming
        timestamp = str(int(time.time()))

        # Sanitize the user-provided filename
        sanitized_filename = sanitize_filename(filename)

        # Get output directory
        output_root = os.path.join(os.getcwd(), "ComfyUI", "output")

        # Try multiple directory locations
        directories_to_try = [
            output_root,
            os.path.join(os.getcwd(), "output"),
            os.path.join(tempfile.gettempdir(), "comfyui_output"),
            os.path.dirname(__file__),
        ]

        output_path = None
        for test_path in directories_to_try:
            if safe_ensure_directory(test_path):
                output_path = test_path
                break

        if output_path is None:
            log("Failed to create any output directory")
            return {"ui": {"error": "Cannot create output directory"}}

        saved_count = 0
        failed_count = 0

        for i, image in enumerate(images):
            try:
                # Convert tensor to PIL with robust error handling
                pil_img = tensor_to_pil(image, "SaveImageCompressed")

                # Generate unique save path
                # If user provided a specific filename, use it (with duplicate handling)
                # If no filename provided (empty or default), use timestamp-based naming
                if sanitized_filename and sanitized_filename not in ["output.png", ""]:
                    save_path, save_name = get_unique_save_path(output_path, sanitized_filename)
                else:
                    # Generate timestamp-based unique filename for default case
                    rand = os.urandom(4).hex()
                    save_name = f"{timestamp}_{i}_{rand}.png"
                    save_path = os.path.join(output_path, save_name)
                    # Ensure uniqueness even for timestamp names
                    while os.path.exists(save_path):
                        rand = os.urandom(4).hex()
                        save_name = f"{timestamp}_{i}_{rand}.png"
                        save_path = os.path.join(output_path, save_name)

                # Try to compress and save
                try:
                    compressed = compress_image(pil_img, quality_min, quality_max)
                    with safe_open_file(save_path, "wb") as f:
                        f.write(compressed)
                except Exception as e:
                    log(f"Compression failed, saving uncompressed: {e}")
                    # Fallback to direct PIL save
                    try:
                        pil_img.save(save_path)
                    except Exception as save_error:
                        log(f"Failed to save: {save_error}")
                        failed_count += 1
                        continue

                saved_count += 1
                log(f"Saved: {save_name}")

                # Update progress bar
                if pbar:
                    try:
                        pbar.update_absolute(i, images.shape[0], ("PNG", pil_img, None))
                    except Exception:
                        pass

            except Exception as e:
                log(f"Failed to process image {i}: {e}")
                failed_count += 1
                continue

        return {}


NODE_CLASS_MAPPINGS = {
    "SaveImageCompressed": SaveImageCompressed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageCompressed": "Save Compressed Image ",
}