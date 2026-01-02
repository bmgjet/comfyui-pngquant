SaveImageCompressed (ComfyUI Node)

A robust, ComfyUI output node that saves images as compressed PNGs using pngquant.

--------------------------------------------------
FEATURES

- PNG compression via pngquant
- Automatic pngquant download (Windows & Linux)
- Graceful fallback to uncompressed PNG if compression fails
- Handles NaNs, invalid tensors, unexpected shapes
- Safe filename sanitization (prevents path traversal)
- Automatic duplicate filename handling
- Multiple output directory fallbacks
- Never crashes ComfyUI execution
- Works with batch images
- Progress bar support

--------------------------------------------------
WHAT THIS NODE DOES

- Converts ComfyUI image tensors to PIL images safely
- Compresses PNGs using pngquant with a quality range
- Saves images to a valid output directory
- Ensures output always exists, even under failure conditions

If anything goes wrong:
- pngquant missing → saves uncompressed PNG
- tensor invalid → saves fallback image
- directory not writable → tries another location
- filename collision → auto-increments name

--------------------------------------------------
NODE INFORMATION

Node Name:        SaveImageCompressed
Category:         pngquant
Type:             Output Node
Batch Support:    Yes

--------------------------------------------------
INPUTS

images
Type: IMAGE
Description: Image tensor(s) from ComfyUI

filename
Type: STRING
Description: Output filename (optional)

quality_min
Type: INT (0–100)
Description: Minimum pngquant quality

quality_max
Type: INT (0–100)
Description: Maximum pngquant quality

Notes:
- If filename is empty or default, a timestamp-based filename is generated.
- Duplicate filenames automatically get _1, _2, etc.

--------------------------------------------------
OUTPUT LOCATIONS (FALLBACK ORDER)

1. ComfyUI/output
2. ./output
3. System temp directory (comfyui_output)
4. Node directory (last resort)

--------------------------------------------------
COMPRESSION DETAILS

- Uses pngquant for high-quality PNG compression
- Quality range controlled by quality_min and quality_max
- 30-second timeout protection
- pngquant is automatically downloaded if missing

Supported platforms:
- Windows
- Linux

If pngquant fails for any reason, the image is still saved.

--------------------------------------------------
SAFETY AND STABILITY

This node is intentionally defensive:
- Sanitizes filenames
- Prevents directory traversal
- Handles corrupted tensors
- Never raises uncaught exceptions
- Never blocks ComfyUI execution

Worst-case scenario:
A valid PNG is still saved.

--------------------------------------------------
INSTALLATION

1. Copy this node file into your ComfyUI custom_nodes directory
2. Restart ComfyUI
3. Look for "Save Compressed Image" under the pngquant category

--------------------------------------------------
EXAMPLE WORKFLOW

KSampler
  ↓
VAE Decode
  ↓
SaveImageCompressed

--------------------------------------------------
NODE REGISTRATION

NODE_CLASS_MAPPINGS:
- SaveImageCompressed → SaveImageCompressed

NODE_DISPLAY_NAME_MAPPINGS:
- SaveImageCompressed → Save Compressed Image

--------------------------------------------------
KNOWN BEHAVIOR (BY DESIGN)

- If compression fails, the image is saved uncompressed
- If tensor format is unexpected, the image is auto-converted
- If filename is invalid, it is sanitized automatically

This behavior is intentional to maximize reliability.

--------------------------------------------------
LICENSE

Free to use in personal ComfyUI workflows.
pngquant is distributed under its own license.

--------------------------------------------------
CREDITS

- pngquant for PNG compression
- ComfyUI for the node framework
