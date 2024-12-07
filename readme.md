# nanojpeg

This is a simple JPEG Encoder/Decoder, written in Python, mostly for educational purposes.[^1]

It supports baseline sequential DCT with Huffman coding and 4:4:4, 4:2:2 or 4:2:0 subsampling.[^2]

It also implements the F4 algorithm for encoding a payload into the quantized AC coefficients.[^3]

No Copyright () 2024 Robert Luxemburg. No rights reserved. This code is in the Public Domain.[^4]

![nanojpeg](nanojpeg.jpg)

### Installation:

```
git clone https://github.com/rolux/nanojpeg.git
cd nanojpeg
pip install -r requirements.txt
```

### Usage:

```python
# Test encoder and decoder
from nanojpeg import jpeg
original = [
    [(x, x * y // 256, y) for x in range(256)]
    for y in range(256)
]
jpeg.encode("test q=75 s=420.jpg", original)
image = jpeg.decode("test q=75 s=420.jpg")
jpeg.encode("test q=75 s=420 decoded and re-encoded.jpg", image)

# Try different quality and subsampling settings
jpeg.encode("test q=50 s=422.jpg", original, quality=50, subsampling="4:2:2")
jpeg.encode("test q=95 s=444.jpg", original, quality=95, subsampling="4:4:4")

# Export the quantization tables
image, metadata = jpeg.decode("test q=50 s=422.jpg", return_metadata=True)
print(metadata["qt"])
image, metadata = jpeg.decode("test q=95 s=444.jpg", return_metadata=True)
print(metadata["qt"])

# Work with an existing image file
import numpy as np
from PIL import Image
original = np.array(Image.open("nanojpeg.jpg"))
jpeg.encode("nanojpeg re-encoded.jpg", original)

# Encode and decode a payload
original = jpeg.decode("nanojpeg re-encoded.jpg")
payload = open("test q=75 s=420.jpg", "rb").read()
jpeg.encode("nanojpeg re-encoded with payload.jpg", original, payload=payload)
image, payload = jpeg.decode("nanojpeg re-encoded with payload.jpg", return_payload=True)
open("nanojpeg payload.jpg", "wb").write(payload)

# Plot histograms of AC coefficients
import matplotlib.pyplot as plt
for filename in ("nanojpeg re-encoded.jpg", "nanojpeg re-encoded with payload.jpg"):
    image, metadata = jpeg.decode(filename, return_metadata=True)
    keys = list(metadata["ac"].keys())
    # insert zero for missing keys, take the square root to squash values, leave out key zero
    ac = [metadata["ac"].get(key, 0)**.5 for key in range(keys[0], keys[-1] + 1) if key != 0]
    plt.plot(ac)
    plt.axis("off")
    plt.savefig(filename.replace(".jpg", " (histogram).png"))
    plt.close()

# Extract the full nanojpeg source code from nanojpeg.jpg
image, payload = jpeg.decode("nanojpeg.jpg", return_payload=True)
open("nanojpeg payload.txt", "wb").write(payload)
```

### The JPEG encoding TL;DR:

```
- Convert from RGB to YCbCr
- Shift by -128
- Split into color components
- For each component:
    - Downsample (if needed)
    - Split into blocks of 8x8 pixels
    - Group as MCUs (depending on subsampling)
    - For each MCU:
        - For each block:
            - Apply Discrete Cosine Transform
            - Quantize
            - Encode payload (if present)
            - Zigzag-reorder
            - Encode 1 DC coefficient (Huffman)
            - Encode 63 AC coefficients (Huffman/RLE)
            - Add DC and AC coefficients to scan data
- Pad the scan data with 1 bits to the next full byte
- In the scan data, insert null bytes after FF bytes
- Write markers, metadata segments and scan data

An MCU (Minimum Coded Unit) is a group of blocks covering the same image area.
For example, with 4:2:0 subsampling, it contains 4 Y blocks, 1 Cr block and
1 Cb block. Cr and Cb must be downsampled accordingly.

The scan data is encoded bitwise. Negative values are bit-flipped.
DC coefficient: Hufmann-encoded size in bits, then the difference from the
previous value in the same component.
AC coefficients: Hufmann-encoded (4 bits runlength, 4 bits size in bits),
then the non-zero value. Runlength is the number of preceding zeros.
(15, 0) = 16 zeros, (0, 0) = all zeros until the end of the block.
```

[^1]: There are several other educational python implementations available on GitHub, but most of them turned out to be incomplete, convoluted or confused. This one attempts to be more pythonic, i.e. simple, idiomatic and readable. It depends on numpy (for array manipulation) and scipy (for bicubic interpolation and discrete cosine transform), the rest is pure python.

[^2]: See the [JPEG Specification](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) or [Wikipedia](https://en.wikipedia.org/wiki/JPEG).

[^3]: F4 is described in the paper [F5 - A Steganographic Algorithm](https://digitnet.github.io/assets/pdf/f5-a-steganographic-algorithm-high-capacity-despite-better-steganalysis.pdf).

[^4]: See [Wikipedia](https://en.wikipedia.org/wiki/Public_domain).