import re
import struct
import numpy as np
from scipy import fftpack, interpolate


class JPEG():

    """
    This is a simple JPEG Encoder/Decoder, written in Python, mostly for educational purposes.
    Supports baseline sequential DCT with Huffman coding and 4:4:4, 4:2:2 or 4:2:0 subsampling.
    Also implements the F4 algorithm for encoding a payload into the quantized AC coefficients.
    No Copyright () 2024 Robert Luxemburg. No rights reserved. This code is in the Public Domain.
    """

    def __init__(self):
        self._marker_names = {
            0xFFC0: "SOF (Start of frame)",
            0xFFC4: "DHT (Define Huffman table)",
            0xFFD8: "SOI (Start of image)",
            0xFFD9: "EOI (End of image)",
            0xFFDA: "SOS (Start of scan)",
            0xFFDB: "DQT (Define quantization table)",
            0xFFDD: "DRI (Define restart interval)"
        }
        self._marker_names.update({
            0xFFD0 + i: f"RST{i} (Restart)" for i in range(8)
        })
        self._marker_names.update({
            0xFFE0 + i: f"APP{i} (Application)" for i in range(16)
        })
        self._marker_values = {v: k for k, v in self._marker_names.items()}
        # Quantization tables (luminance), from PIL/libjpeg
        self._qty = {}
        # Quantization tables (chrominance), from PIL/libjpeg
        self._qtc = {}
        qt = np.load("qt.npy") # (19, 2, 8, 8)
        for idx, tables in enumerate(qt):
            quality = (idx + 1) * 5
            self._qty[quality], self._qtc[quality] = tables
        # Huffman table (luminance DC), from the JPEG Specification
        self._htydc = (
            [], [0], [1, 2, 3, 4, 5], [6], [7], [8], [9], [10], [11], [], [], [], [], [], [], []
        )
        # Huffman table (luminance AC), from the JPEG Specification
        self._htyac = (
            [], [1, 2], [3], [0, 4, 17], [5, 18, 33], [49, 65], [6, 19, 81, 97], [7, 34, 113], [20,
            50, 129, 145, 161], [8, 35, 66, 177, 193], [21, 82, 209, 240], [36, 51, 98, 114], [],
            [], [130], [9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57,
            58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102,
            103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135,
            136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166,
            167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197,
            198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227,
            228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
        )
        # Huffman table (chrominance DC), from the JPEG Specification
        self._htcdc = (
            [], [0, 1, 2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [], [], [], [], []
        )
        # Huffman table (chrominance AC), from the JPEG Specification
        self._htcac = (
            [], [0, 1], [2], [3, 17], [4, 5, 33, 49], [6, 18, 65, 81], [7, 97, 113], [19, 34, 50,
            129], [8, 20, 66, 145, 161, 177, 193], [9, 35, 51, 82, 240], [21, 98, 114, 209], [10,
            22, 36, 52], [], [225], [37, 241], [23, 24, 25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56,
            57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101,
            102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132, 133,
            134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164,
            165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195,
            196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226,
            227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248, 249, 250]
        )
        # Zigzag order
        self._zz = (
             0,  1,  5,  6, 14, 15, 27, 28,
             2,  4,  7, 13, 16, 26, 29, 42,
             3,  8, 12, 17, 25, 30, 41, 43,
             9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        )

    class _BitWriter():
        """Allows for bit-wise accumulation of data"""
        def __init__(self):
            self._data = b""
            self._bits = []
        def _bits_to_int(self, bits):
            val = 0
            for bit in bits:
                val = (val << 1) | bit
            return val
        def write(self, bits, stuff=False):
            self._bits += bits
            while len(self._bits) >= 8:
                self._data += bytes([self._bits_to_int(self._bits[:8])])
                if stuff and self._data[-1:] == b"\xFF":
                    self._data += b"\x00"
                self._bits = self._bits[8:]
            return self
        def pad_to_next_byte(self, padding, stuff=False):
            if mod := len(self._bits) % 8:
                self.write((8 - mod) * [padding], stuff)
            return self
        def get_data(self):
            return self._data

    class _BitReader():
        """Allows for bit-wise consumption of data"""
        def __init__(self, data):
            self._data = data
            self._index = 0
        def read(self, n=None):
            remaining = len(self._data) * 8 - self._index
            if n is None:
                n = remaining
            if n > remaining:
                raise ValueError(f"Tried to read {n} bit(s) from remaining {remaining} bit(s)")
            bits = [
                self._data[i//8] >> (7 - i % 8) & 1
                for i in range(self._index, self._index + n)
            ]
            self._index += n
            return bits
        def seek(self, i):
            self._index += i
            if self._index < 0 or self._index > 8 * len(self._data) - 1:
                raise ValueError(f"Tried to seek to index {self._index}, which is out of bounds.")
            return self
        def seek_to_next_byte(self):
            if mod := self._index % 8:
                self.seek(8 - mod)
            return self

    def _bits_to_int(self, bits):
        is_negative = bits[0] == 0
        if is_negative:
            bits = [1 - bit for bit in bits]
        val = 0
        for bit in bits:
            val = (val << 1) | bit
        return val * -1 if is_negative else val

    def _int_to_bits(self, val):
        is_negative = val < 0
        bits = [int(bit) for bit in bin(val)[2 + is_negative:]]
        return [1 - bit for bit in bits] if is_negative else bits

    def _concatenate(self, arrs, wh):
        """Arrange w*h arrays of the same shape in a w-by-h grid"""
        w, h = wh
        rows = np.array([np.hstack(arrs[i * w:(i + 1) * w]) for i in range(h)])
        return np.vstack(rows)

    def _pad(self, arr, block_shape):
        """Pad array to a multiple of block shape by repeating the last element"""
        for axis in (1, 0):
            if mod := arr.shape[axis] % block_shape[axis]:
                n = block_shape[axis] - mod
                arr = np.repeat(arr, (arr.shape[axis] - 1) * [1] + [1 + n], axis=axis)
        return arr

    def _resample(self, arr, sf):
        """Resample array according to horizontal and vertical scaling factor"""
        h, w = arr.shape
        sfh, sfv = sf
        # This is correct, x/y are not coordinates.
        x = np.array(range(h))
        y = np.array(range(w))
        f = interpolate.RectBivariateSpline(x, y, arr)
        x_new = np.linspace(0, h - 1, int(sfv * h))
        y_new = np.linspace(0, w - 1, int(sfh * w))
        return f(x_new, y_new)

    def _zigzag(self, arr):
        arr = arr.reshape((64,))
        arr = np.array([arr[self._zz.index(i)] for i in range(64)])
        return arr.reshape((8, 8))

    def _dezigzag(self, arr):
        arr = arr.reshape((64,))
        arr = np.array([arr[self._zz[i]] for i in range(64)])
        return arr.reshape((8, 8))

    def _ycbcr(self, rgb):
        r, g, b = np.transpose(rgb, (2, 0, 1))
        return np.dstack((
             0.299  * r + 0.587  * g + 0.114  * b,
            -0.1687 * r - 0.3313 * g + 0.5    * b + 128,
             0.5    * r - 0.4187 * g - 0.0813 * b + 128
        ))

    def _rgb(self, ycbcr):
        y, cb, cr = np.transpose(ycbcr, (2, 0, 1))
        return np.dstack((
            y                        + 1.402   * (cr - 128),
            y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128),
            y + 1.772   * (cb - 128)
        ))

    def _dct(self, arr):
        return fftpack.dct(fftpack.dct(arr.T, norm="ortho").T, norm="ortho")

    def _idct(self, arr):
        return fftpack.idct(fftpack.idct(arr.T, norm="ortho").T, norm="ortho")

    def _parse_huffman_table(self, table, mode):
        assert(mode in ("decode", "encode"))
        data = {}
        code = 0
        for idx, vals in enumerate(table):
            length = idx + 1
            for val in vals:
                code_bin = ("0" + bin(code)[2:])[-length:]
                if mode == "decode":
                    data[code_bin] = val
                else:
                    data[val] = code_bin
                code += 1
            code *= 2
        return data

    def _huffman_encode(self, val, table):
        if val not in table:
            raise ValueError("No Huffman code for value {val}.")
        return [int(c) for c in table[val]]

    def _huffman_decode(self, bitreader, table, max_len=16):
        code = ""
        while len(code) <= max_len:
            code += str(bitreader.read(1)[0])
            if code in table:
                return table[code]
        raise ValueError(f"Huffman code {code} is longer than {max_len}.")

    def _encode_dc(self, dc, bitwriter, table):
        """
        1 entry: [ huffman(len(bits(dc))) ] [ bits(dc) ]
        Negative dc values are bit-flipped.
        """
        dc = [] if dc == 0 else self._int_to_bits(dc)
        size = self._huffman_encode(len(dc), table)
        bitwriter.write(size + dc, stuff=True)

    def _decode_dc(self, bitreader, table):
        size = self._huffman_decode(bitreader, table)
        return 0 if size == 0 else self._bits_to_int(bitreader.read(size))

    def _encode_ac(self, ac, bitwriter, table):
        """
        63 entries: [ huffman(4 bits runlength, 4 bits len(bits(ac))) ] [ bits(ac) ]
        Negative ac values are bit-flipped. Runlength is the number of preceding zeros.
        (15, 0) is ZRL (Zero runlength) -> 16 zeros
        ( 0, 0) is EOB (End of Block)   -> All zeros until end of block
        """
        def runlength_encode(run, size, val):
            runsize = self._huffman_encode(run * 16 + size, table)
            bitwriter.write(runsize + val, stuff=True)
        nonzero = np.nonzero(ac)[0]
        eob = 0 if nonzero.size == 0 else np.max(nonzero) + 1
        run = 0
        for idx, val in enumerate(ac):
            if idx == eob:
                runlength_encode(0, 0, []) # EOB
                break
            if val == 0:
                run += 1
                if run == 16:
                    runlength_encode(15, 0, []) # ZRL
                    run = 0
            else:
                val = self._int_to_bits(val)
                size = len(val)
                runlength_encode(run, size, val)
                run = 0

    def _decode_ac(self, bitreader, table, n=63):
        ac = []
        while len(ac) < n:
            runsize = self._huffman_decode(bitreader, table)
            run, size = runsize >> 4, runsize & 15
            if run == 0 and size == 0: # EOB
                ac += (n - len(ac)) * [0]
            elif run == 15 and size == 0: # ZRL
                ac += 16 * [0]
            else:
                ac += run * [0] + [self._bits_to_int(bitreader.read(size))]
        return ac

    def _encode_payload(self, block, payload_reader):
        """
        Encodes the payload bitwise into all non-zero AC coefficients, as value % 2
        for positive and 1 - value % 2 for negative values, by decreasing (if needed)
        the absolute value by 1. If the result is zero, the encoding is discarded.
        """
        block = block.reshape((64,))
        for idx, val in enumerate(block):
            if idx > 0 and val != 0:
                mod_bit = val % 2 if val > 0 else 1 - val % 2
                try:
                    payload_bit = payload_reader.read(1)[0]
                except ValueError: # encoding complete
                    continue
                if mod_bit != payload_bit:
                    block[idx] += (1, -1)[val > 0]
                    if block[idx] == 0:
                        payload_reader.seek(-1)
        return block.reshape((8, 8))

    def _decode_payload(self, block, payload_writer):
        for idx, val in enumerate(block.reshape((64,))):
            if idx > 0 and val != 0:
                payload_writer.write([val % 2 if val > 0 else 1 - val % 2])

    def encode(
        self, filename, image_data, quality=75, subsampling="4:2:0",
        restart_interval=None, payload=None
    ):

        image_data = np.array(image_data)
        image_height, image_width = image_data.shape[:2]
        if quality < 5 or quality > 95 or quality % 5:
            raise ValueError("Quality must be a multiple of 5 between 5 and 95.")
        if subsampling not in ("4:4:4", "4:2:2", "4:2:0"):
            raise ValueError("Subsampling must be 4:4:4, 4:2:2 or 4:2:0.")
        h = {"4:4:4": 1, "4:2:2": 2, "4:2:0": 2}[subsampling]
        v = {"4:4:4": 1, "4:2:2": 1, "4:2:0": 2}[subsampling]
        cmeta = {
            #  scaling factors h/v, qnt. table, huffman tables dc/ac
            1: {"sfh": h, "sfv": v, "qtid": 0, "htdc": 0, "htac": 0}, # Y
            2: {"sfh": 1, "sfv": 1, "qtid": 1, "htdc": 1, "htac": 1}, # Cb
            3: {"sfh": 1, "sfv": 1, "qtid": 1, "htdc": 1, "htac": 1}  # Cr
        }
        mcu_shape = {"4:4:4": (8, 8), "4:2:2": (8, 16), "4:2:0": (16, 16)}[subsampling]
        if payload:
            payload = struct.pack(">HH", 0xFFE0, len(payload) + 2) + payload
            payload_reader = self._BitReader(payload)

        # Pad
        image_data = self._pad(image_data, mcu_shape)
        # RGB to YCbCr
        image_data = self._ycbcr(image_data)
        # Shift
        image_data = image_data.astype(np.int32) - 128
        # Split into color components
        cids = cmeta.keys()
        cdata = {cid: image_data[:, :, i] for i, cid in enumerate(cids)}
        # Split into blocks
        blocks = {cid: [] for cid in cids}
        for cid, meta in cmeta.items():
            if cid == 1 and subsampling == "4:2:0":
                blocks[cid] = [
                    cdata[cid][y+dy:y+dy+8, x+dx:x+dx+8]
                    for y in range(0, cdata[cid].shape[0], 16)
                    for x in range(0, cdata[cid].shape[1], 16)
                    for dy in (0, 8)
                    for dx in (0, 8)
                ]
            else:
                if cid > 1 and subsampling != "4:4:4":
                    sfh, sfv = cmeta[1]["sfh"], cmeta[1]["sfv"]
                    cdata[cid] = self._resample(cdata[cid], (1 / sfh, 1 / sfv))
                blocks[cid] = [
                    cdata[cid][y:y+8, x:x+8]
                    for y in range(0, cdata[cid].shape[0], 8)
                    for x in range(0, cdata[cid].shape[1], 8)
                ]
        # Encode scan data
        bitwriter = self._BitWriter()
        prev_dc = {cid: None for cid in cids}
        htydc = self._parse_huffman_table(self._htydc, mode="encode")
        htyac = self._parse_huffman_table(self._htyac, mode="encode")
        htcdc = self._parse_huffman_table(self._htcdc, mode="encode")
        htcac = self._parse_huffman_table(self._htcac, mode="encode")
        n_mcus = image_data.shape[0] // mcu_shape[0] * image_data.shape[1] // mcu_shape[1]

        for i_mcu in range(n_mcus):
            for cid, meta in cmeta.items():
                n_blocks = meta["sfh"] * meta["sfv"]
                for i_block in range(n_blocks):
                    block = blocks[cid].pop(0)
                    # DCT
                    block = self._dct(block)
                    # Quantize
                    qt = (self._qty[quality], self._qtc[quality])[meta["qtid"]]
                    block = (block / qt).round().astype(np.int32)
                    # Encode payload
                    if payload:
                        block = self._encode_payload(block, payload_reader)
                    # Zigzag
                    block = self._zigzag(block).reshape((64,))
                    dc, ac = block[0], block[1:]
                    # Huffman (DC)
                    curr_dc = dc
                    if prev_dc[cid] is not None:
                        dc = dc - prev_dc[cid]
                    prev_dc[cid] = curr_dc
                    ht = (htydc, htcdc)[meta["htdc"]]
                    self._encode_dc(dc, bitwriter, ht)
                    # Huffman/RLE (AC)
                    ht = (htyac, htcac)[meta["htdc"]]
                    self._encode_ac(ac, bitwriter, ht)
            if (
                restart_interval and i_mcu < n_mcus - 1
                and i_mcu % restart_interval == restart_interval - 1
            ):
                bitwriter.pad_to_next_byte(1, stuff=True)
                restart_counter = i_mcu // restart_interval % 8
                bitwriter.write(self._int_to_bits(0xFFD0 + restart_counter))
                prev_dc = {cid: None for cid in cids}

        scan_data = bitwriter.pad_to_next_byte(1).get_data()

        with open(filename, "wb") as f:

            def write_segment(marker_name, data=None, scan_data=None):
                marker_value = self._marker_values[marker_name]
                if data is None:
                    f.write(struct.pack(">H", marker_value))
                elif scan_data:
                    f.write(struct.pack(">H", marker_value) + data + scan_data)
                else:
                    f.write(struct.pack(">HH", marker_value, len(data) + 2) + data)

            write_segment("SOI (Start of image)")

            marker_name = "DQT (Define quantization table)"
            for i, qt in enumerate((self._qty[quality], self._qtc[quality])):
                # Using struct.pack(">B", x) rather than bytes([x]) for consistency and readablility
                precision, table_id = 0, i
                data = struct.pack(">B", precision * 16 + table_id)
                data += struct.pack(">" + 64 * "B", *qt.reshape((64,)))
                write_segment(marker_name, data)

            marker_name = "SOF (Start of frame)"
            precision = 8
            n_components = len(cmeta)
            data = struct.pack(">BHHB", precision, image_height, image_width, n_components)
            for cid, meta in cmeta.items():
                data += struct.pack(">BBB", cid, meta["sfh"] * 16 + meta["sfv"], meta["qtid"])
            write_segment(marker_name, data)

            marker_name = "DHT (Define Huffman table)"
            for i, ht in enumerate((self._htydc, self._htyac, self._htcdc, self._htcac)):
                table_class = i % 2
                table_id = i // 2
                data = struct.pack(">B", table_class * 16 + table_id)
                n_codes = [len(vals) for vals in ht]
                data += struct.pack(">" + 16 * "B", *n_codes)
                for i, n in enumerate(n_codes):
                    data += struct.pack(">" + n * "B", *ht[i])
                write_segment(marker_name, data)

            marker_name = "DRI (Define restart interval)"
            if restart_interval:
                data = struct.pack(">H", restart_interval)
                write_segment(marker_name, data)

            marker_name = "SOS (Start of scan)"
            header_length = 6 + 2 * n_components
            data = struct.pack(">HB", header_length, n_components)
            for i in range(n_components):
                cid = i + 1
                table_dc = 0 if i == 0 else 1
                table_ac = 0 if i == 0 else 1
                data += struct.pack(">BB", cid, table_dc * 16 + table_ac)
            start_of_selection = 0
            end_of_selection = 63
            bit_position_high = 0
            bit_position_low = 0
            data += struct.pack(
                ">BBB", start_of_selection, end_of_selection,
                bit_position_high * 16 + bit_position_low
            )
            write_segment(marker_name, data, scan_data)

            write_segment("EOI (End of image)")

            if payload:
                remaining = len(payload_reader.read())
                if remaining:
                    encoded = len(payload) - remaining // 8 - 1
                    raise ValueError(
                        f"Payload too long. Could only encode {encoded} of {len(payload)} bytes."
                    )


    def decode(self, filename, return_metadata=False, return_payload=False):

        qt = {} # Quantization tables
        ht = {} # Huffman tables
        restart_interval = None
        if return_metadata:
            metadata = {"markers": [], "qt": {}, "ht": {}, "ac": [], "nz": []}
        if return_payload:
            payload_writer = self._BitWriter()

        with open(filename, "rb") as f:

            def read(pattern):
                split = None
                if "(" in pattern:
                    split = pattern.index("(") - 1
                    pattern = pattern.replace("(", "").replace(")", "")
                length = pattern.count("B") + 2 * pattern.count("H") + 4 * pattern.count("I")
                data = f.read(length)
                if len(data) < length:
                    return None
                data = struct.unpack(pattern, data)
                if split is not None:
                    data = list(data)
                    data[split] = data[split] >> 4, data[split] & 15
                    data = tuple(data)
                return data

            marker_name = self._marker_names.get(read(">H")[0])
            if marker_name != "SOI (Start of image)":
                raise ValueError("No start of image marker.")
            if return_metadata:
                metadata["markers"].append(marker_name)

            while True:

                marker_value = read(">H")[0]
                if marker_value is None:
                    raise ValueError("Premature end of image.")
                marker_name = self._marker_names.get(marker_value)
                if marker_name is None:
                    if marker_value > 0xFFC0 and marker_value < 0xFFD0:
                        raise NotImplementedError(f"Unsupported encoding {marker_value:04X}.")
                    else:
                        raise ValueError(f"Unknown marker {marker_value:04X}.")
                if return_metadata:
                    metadata["markers"].append(marker_name)
                if marker_name != "SOS (Start of scan)":
                    length = read(">H")[0] - 2

                if marker_name.startswith("APP"):
                    data = f.read(length)

                elif marker_name == "DQT (Define quantization table)":
                    precision, table_id = read(">(B)")[0]
                    data = f.read(length - 1)
                    qt[table_id] = np.array([int(v) for v in data]).reshape((8, 8))
                    if return_metadata:
                        metadata["qt"][table_id] = qt[table_id].tolist()

                elif marker_name == "SOF (Start of frame)":
                    precision, image_height, image_width, n_components = read(">BHHB")
                    if precision != 8:
                        raise ValueError(
                            f"Only 8-bit (not {precision}-bit) precision is supported."
                        )
                    if n_components != 3:
                        raise ValueError(
                            f"Only 3 (not {n_components}) color channels are supported."
                        )
                    cmeta = {}
                    for i in range(n_components):
                        cid, (sfh, sfv), qtid = read(">B(B)B")
                        cmeta[cid] = {
                            "sfh": sfh,  # Sampling factor horizontal
                            "sfv": sfv,  # Sampling factor vertical
                            "qtid": qtid # Quantization table ID
                        }
                        if i == 0:
                            subsampling = ("4:4:4", "4:2:2", "4:2:0")[sfh + sfv - 2]

                elif marker_name == "DHT (Define Huffman table)":
                    table_class, table_id = read(">(B)")[0]
                    n_codes = read(">" + 16 * "B")
                    key = (table_class, table_id)
                    ht[key] = [list(read(">" + n * "B")) for n in n_codes]
                    if return_metadata:
                        metadata["ht"][str(key)] = ht[key] # str for valid json
                    ht[key] = self._parse_huffman_table(ht[key], mode="decode")

                elif marker_name == "DRI (Define restart interval)":
                    restart_interval = read(">H")[0]

                elif marker_name == "SOS (Start of scan)":
                    header_length, n_components = read(">HB")
                    if n_components != 3:
                        raise ValueError(
                            f"Only 3 (not {n_components}) color channels are supported."
                        )
                    for i in range(n_components):
                        cid, (table_dc, table_ac) = read(">B(B)")
                        cmeta[cid].update({
                            "htdc": table_dc,
                            "htac": table_ac
                        })
                    selection_start, selection_end, (bit_pos_high, bit_pos_low) = read(">BB(B)")
                    scan_data = f.read()[:-2]
                    f.seek(-2, 1)
                    break

            marker_name = self._marker_names.get(read(">H")[0])
            if marker_name != "EOI (End of image)":
                raise ValueError("No end of image marker.")
            if return_metadata:
                metadata["markers"].append(marker_name)

        # Decode scan data
        parts = re.split(b"(\xFF[\xD0-\xD7])", scan_data)
        scan_data = b"".join(re.sub(b"\xFF\x00", b"\xFF", part) for part in parts)
        bitreader = self._BitReader(scan_data)
        cids = cmeta.keys()
        prev_dc = {cid: None for cid in cids}
        mcus = {cid: [] for cid in cids}
        mcu_shape = {"4:4:4": (8, 8), "4:2:2": (8, 16), "4:2:0": (16, 16)}[subsampling]
        n_mcus = int(np.ceil(image_height / mcu_shape[0]) * np.ceil(image_width / mcu_shape[1]))

        for i_mcu in range(n_mcus):
            for cid, meta in cmeta.items():
                blocks = []
                n_blocks = meta["sfh"] * meta["sfv"]
                for i_block in range(n_blocks):
                    # Huffman (DC)
                    dc = self._decode_dc(bitreader, ht[(0, meta["htdc"])])
                    if prev_dc[cid] is not None:
                        dc = prev_dc[cid] + dc
                    prev_dc[cid] = dc
                    # Huffman/RLE (AC)
                    ac = self._decode_ac(bitreader, ht[(1, meta["htac"])])
                    if return_metadata:
                        metadata["ac"] += list(ac)
                        metadata["nz"] += [np.count_nonzero(ac)]
                    block = np.array([dc] + ac).reshape((8, 8))
                    # Dezigzag
                    block = self._dezigzag(block)
                    # Decode Payload
                    if return_payload:
                        self._decode_payload(block, payload_writer)
                    # Dequantize
                    block = (block * qt[meta["qtid"]])
                    # Inverse DCT
                    block = self._idct(block)
                    blocks.append(block)
                # Concatenate to MCU block
                if n_blocks == 1:
                    mcus[cid].append(blocks[0])
                elif n_blocks == 2:
                    mcus[cid].append(np.hstack(blocks))
                elif n_blocks == 4:
                    mcus[cid].append(self._concatenate(blocks, (2, 2)))
            if (
                restart_interval and i_mcu < n_mcus - 1
                and i_mcu % restart_interval == restart_interval - 1
            ):
                bitreader.seek_to_next_byte()
                restart_marker = self._bits_to_int(bitreader.read(16))
                restart_counter = i_mcu // restart_interval % 8
                expected = 0xFFD0 + restart_counter
                if restart_marker != expected:
                    if restart_marker < 0xFFD0 or restart_marker > 0xFFD7:
                        raise ValueError("Missing restart marker.")
                    else:
                        raise ValueError(
                            f"Incorrect restart marker. Expected {expected:04X}, "
                            f"got {restart_marker:04X}."
                        )
                if return_metadata:
                    metadata["markers"].insert(-1, self._marker_names[restart_marker])
                prev_dc = {cid: None for cid in cids}

        if return_metadata:
            metadata["ac"] = dict(zip(*np.unique(metadata["ac"], return_counts=True)))
            metadata["nz"] = dict(zip(*np.unique(metadata["nz"], return_counts=True)))
        # Concatenate blocks
        cdata = {}
        for cid, meta in cmeta.items():
            w, h = image_width // mcu_shape[1], image_height // mcu_shape[0]
            cdata[cid] = self._concatenate(mcus[cid], (w, h))
            if cid > 1 and subsampling != "4:4:4":
                # Upsample
                sfh, sfv = cmeta[1]["sfh"], cmeta[1]["sfv"]
                cdata[cid] = self._resample(cdata[cid], (sfh, sfv))
        # Join color components
        image_data = np.dstack([cdata[cid] for cid in cids])
        # Unshift
        image_data = image_data + 128
        # YCbCr to RGB
        image_data = self._rgb(image_data)
        image_data = image_data.round().clip(0, 255).astype(np.uint8)
        # Crop
        image_data = image_data[:image_height, :image_width]
        # Padding
        remaining_bits = bitreader.read()
        if len(remaining_bits) > 7:
            raise ValueError("Scan data has more than 7 bits of padding.")
        if any(bit == 0 for bit in remaining_bits):
            raise ValueError("Scan data has zero-bits in padding.")

        if return_payload:
            payload = payload_writer.get_data()
            payload_marker = struct.unpack(">H", payload[:2])[0]
            if payload_marker != 0xFFE0:
                raise ValueError(f"Unknown payload marker {payload_marker:04X}")
            payload_length = struct.unpack(">H", payload[2:4])[0] - 2
            payload = payload[4:4+payload_length]
            return (image_data, metadata, payload) if return_metadata else image_data, payload

        return (image_data, metadata) if return_metadata else image_data


jpeg = JPEG()