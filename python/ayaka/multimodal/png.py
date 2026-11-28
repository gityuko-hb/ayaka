"""Pure-Python PNG codec, data-URL parser, and synthetic screenshot generator."""

from __future__ import annotations

import base64
import binascii
import struct
import zlib
from typing import Final

from ayaka.core.errors import InvalidRequestError
from ayaka.core.multimodal import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, ImageInput

__all__ = [
    "decode_base64_image",
    "decode_data_url_image",
    "decode_png_rgb",
    "encode_png_rgb",
    "make_ide_screenshot_png",
    "read_png_dimensions",
]

_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_ADAM7_PASSES: Final = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)
# Max raw inflated bytes allowed during decode (RGBA 16-bit max pixels + filter bytes).
_MAX_INFLATED_BYTES: Final[int] = MAX_IMAGE_PIXELS * 8 + 65536


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter_scanlines(raw: bytes, width: int, height: int, bpp: int, stride: int) -> bytearray:
    expected = height * (1 + stride)
    if len(raw) != expected:
        raise InvalidRequestError(
            f"corrupt PNG scanline data: expected {expected} bytes, got {len(raw)}"
        )
    out = bytearray(height * stride)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        row_start = y * stride
        prev_start = (y - 1) * stride
        for x in range(stride):
            val = raw[pos + x]
            a = out[row_start + x - bpp] if x >= bpp else 0
            b = out[prev_start + x] if y > 0 else 0
            c = out[prev_start + x - bpp] if (y > 0 and x >= bpp) else 0
            if ftype == 0:
                rec = val
            elif ftype == 1:
                rec = (val + a) & 0xFF
            elif ftype == 2:
                rec = (val + b) & 0xFF
            elif ftype == 3:
                rec = (val + ((a + b) >> 1)) & 0xFF
            elif ftype == 4:
                rec = (val + _paeth(a, b, c)) & 0xFF
            else:
                raise InvalidRequestError(f"unsupported PNG filter type {ftype}")
            out[row_start + x] = rec
        pos += stride
    return out


def _iter_png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not data.startswith(_PNG_SIGNATURE):
        raise InvalidRequestError("image/png payload is missing a valid PNG header")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidRequestError(
            f"image payload exceeds {MAX_IMAGE_BYTES} byte limit ({len(data)} bytes)"
        )
    chunks: list[tuple[bytes, bytes]] = []
    pos = len(_PNG_SIGNATURE)
    n = len(data)
    while pos + 12 <= n:
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        end = pos + 8 + length
        if end + 4 > n:
            raise InvalidRequestError("truncated PNG chunk")
        payload = data[pos + 8 : end]
        crc_stored = struct.unpack(">I", data[end : end + 4])[0]
        crc_actual = zlib.crc32(tag + payload) & 0xFFFFFFFF
        if crc_stored != crc_actual:
            raise InvalidRequestError(f"corrupt PNG chunk {tag.decode('latin-1', 'replace')} CRC")
        chunks.append((tag, payload))
        pos = end + 4
        if tag == b"IEND":
            break
    if not chunks or chunks[0][0] != b"IHDR" or chunks[-1][0] != b"IEND":
        raise InvalidRequestError("PNG stream missing required IHDR or IEND chunk")
    return chunks


def _parse_ihdr(payload: bytes) -> tuple[int, int, int, int, int]:
    if len(payload) != 13:
        raise InvalidRequestError("invalid PNG IHDR length")
    width, height, bit_depth, color_type, compression, filt, interlace = struct.unpack(
        ">IIBBBBB", payload
    )
    if width <= 0 or height <= 0:
        raise InvalidRequestError(f"invalid PNG dimensions {width}x{height}")
    if width * height > MAX_IMAGE_PIXELS:
        raise InvalidRequestError(
            f"image pixel count {width * height} exceeds {MAX_IMAGE_PIXELS} limit"
        )
    if compression != 0 or filt != 0 or interlace not in (0, 1):
        raise InvalidRequestError("unsupported PNG compression, filter, or interlace method")
    if color_type not in (0, 2, 3, 4, 6):
        raise InvalidRequestError(f"unsupported PNG color type {color_type}")
    valid_depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if bit_depth not in valid_depths[color_type]:
        raise InvalidRequestError(f"invalid PNG bit depth {bit_depth} for color type {color_type}")
    return width, height, bit_depth, color_type, interlace


def read_png_dimensions(data: bytes) -> tuple[int, int]:
    """Return ``(height, width)`` from a PNG header after validation."""
    if not data.startswith(_PNG_SIGNATURE):
        raise InvalidRequestError("image/png payload is missing a valid PNG header")
    if len(data) < 33:
        raise InvalidRequestError("truncated PNG header")
    length, tag = struct.unpack(">I4s", data[8:16])
    if tag != b"IHDR" or length != 13:
        raise InvalidRequestError("invalid PNG IHDR chunk")
    crc_stored = struct.unpack(">I", data[29:33])[0]
    if (zlib.crc32(data[12:29]) & 0xFFFFFFFF) != crc_stored:
        raise InvalidRequestError("corrupt PNG IHDR CRC")
    width, height, *_ = _parse_ihdr(data[16:29])
    return height, width


def _decode_row_to_rgb(
    row: memoryview,
    width: int,
    bit_depth: int,
    color_type: int,
    palette: bytes | None,
) -> bytes:
    out = bytearray(width * 3)
    if color_type == 2 and bit_depth == 8:
        out[:] = row[: width * 3]
        return bytes(out)
    if color_type == 6 and bit_depth == 8:
        for i in range(width):
            j = i * 4
            k = i * 3
            out[k] = row[j]
            out[k + 1] = row[j + 1]
            out[k + 2] = row[j + 2]
        return bytes(out)
    if color_type == 0 and bit_depth == 8:
        for i in range(width):
            v = row[i]
            k = i * 3
            out[k] = v
            out[k + 1] = v
            out[k + 2] = v
        return bytes(out)
    if color_type == 4 and bit_depth == 8:
        for i in range(width):
            v = row[i * 2]
            k = i * 3
            out[k] = v
            out[k + 1] = v
            out[k + 2] = v
        return bytes(out)
    if color_type == 3:
        if palette is None:
            raise InvalidRequestError("indexed PNG is missing PLTE chunk")
        mask = (1 << bit_depth) - 1
        per_byte = 8 // bit_depth
        for i in range(width):
            b = row[i // per_byte]
            shift = (per_byte - 1 - (i % per_byte)) * bit_depth
            idx = ((b >> shift) & mask) * 3
            if idx + 3 > len(palette):
                raise InvalidRequestError("indexed PNG palette index out of range")
            k = i * 3
            out[k : k + 3] = palette[idx : idx + 3]
        return bytes(out)
    if color_type == 0 and bit_depth < 8:
        mask = (1 << bit_depth) - 1
        per_byte = 8 // bit_depth
        scale = 255 // mask
        for i in range(width):
            b = row[i // per_byte]
            shift = (per_byte - 1 - (i % per_byte)) * bit_depth
            v = ((b >> shift) & mask) * scale
            k = i * 3
            out[k] = v
            out[k + 1] = v
            out[k + 2] = v
        return bytes(out)
    # 16-bit grayscale, RGB, Gray+Alpha, RGBA: take high byte per channel
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    step = channels * 2
    for i in range(width):
        src = i * step
        dst = i * 3
        if color_type in (0, 4):
            v = row[src]
            out[dst] = v
            out[dst + 1] = v
            out[dst + 2] = v
        else:
            out[dst] = row[src]
            out[dst + 1] = row[src + 2]
            out[dst + 2] = row[src + 4]
    return bytes(out)


def decode_png_rgb(data: bytes) -> tuple[int, int, bytes]:
    """Decode a PNG byte string into ``(height, width, rgb_bytes)`` (3 bytes/pixel)."""
    chunks = _iter_png_chunks(data)
    width, height, bit_depth, color_type, interlace = _parse_ihdr(chunks[0][1])
    palette: bytes | None = None
    idat_parts: list[bytes] = []
    for tag, payload in chunks[1:]:
        if tag == b"PLTE":
            if len(payload) % 3 != 0 or not payload:
                raise InvalidRequestError("invalid PNG PLTE chunk")
            palette = payload
        elif tag == b"IDAT":
            idat_parts.append(payload)
    if not idat_parts:
        raise InvalidRequestError("PNG stream contains no IDAT data")
    decomp = zlib.decompressobj()
    try:
        inflated = decomp.decompress(b"".join(idat_parts), _MAX_INFLATED_BYTES)
        if not decomp.eof or decomp.unconsumed_tail:
            raise InvalidRequestError("inflated PNG image exceeds maximum decoded size")
    except zlib.error as exc:
        raise InvalidRequestError(f"corrupt PNG compressed data: {exc}") from exc

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    bits_per_pixel = channels * bit_depth
    bpp = max(1, (bits_per_pixel + 7) // 8)

    if interlace == 0:
        stride = (width * bits_per_pixel + 7) // 8
        unfiltered = _unfilter_scanlines(inflated, width, height, bpp, stride)
        rgb = bytearray(width * height * 3)
        mv = memoryview(unfiltered)
        for y in range(height):
            row_rgb = _decode_row_to_rgb(
                mv[y * stride : (y + 1) * stride], width, bit_depth, color_type, palette
            )
            rgb[y * width * 3 : (y + 1) * width * 3] = row_rgb
        return height, width, bytes(rgb)

    # Adam7 interlaced decoding
    rgb = bytearray(width * height * 3)
    pos = 0
    for x0, y0, dx, dy in _ADAM7_PASSES:
        pw = (width - x0 + dx - 1) // dx if width > x0 else 0
        ph = (height - y0 + dy - 1) // dy if height > y0 else 0
        if pw <= 0 or ph <= 0:
            continue
        pstride = (pw * bits_per_pixel + 7) // 8
        plen = ph * (1 + pstride)
        if pos + plen > len(inflated):
            raise InvalidRequestError("truncated interlaced PNG image")
        pass_buf = _unfilter_scanlines(inflated[pos : pos + plen], pw, ph, bpp, pstride)
        pos += plen
        pmv = memoryview(pass_buf)
        for py in range(ph):
            row_rgb = _decode_row_to_rgb(
                pmv[py * pstride : (py + 1) * pstride], pw, bit_depth, color_type, palette
            )
            y = y0 + py * dy
            for px in range(pw):
                x = x0 + px * dx
                dst = (y * width + x) * 3
                src = px * 3
                rgb[dst : dst + 3] = row_rgb[src : src + 3]
    if pos != len(inflated):
        raise InvalidRequestError("extra trailing bytes in interlaced PNG stream")
    return height, width, bytes(rgb)


def encode_png_rgb(width: int, height: int, rgb_bytes: bytes) -> bytes:
    """Encode tight row-major 8-bit RGB bytes into a valid PNG."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions {width}x{height}")
    expected = width * height * 3
    if len(rgb_bytes) != expected:
        raise ValueError(
            f"expected {expected} RGB bytes for {width}x{height}, got {len(rgb_bytes)}"
        )
    stride = width * 3
    scanlines = bytearray(height * (1 + stride))
    for y in range(height):
        dst = y * (1 + stride)
        scanlines[dst] = 0
        scanlines[dst + 1 : dst + 1 + stride] = rgb_bytes[y * stride : (y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(scanlines), level=6)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


_FONT_5X7: Final[dict[str, tuple[str, ...]]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "0": ("01110", "10011", "10101", "10101", "11001", "10001", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "4": ("00110", "01010", "10010", "11111", "00010", "00010", "00010"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
}


def make_ide_screenshot_png(width: int = 1024, height: int = 768) -> bytes:
    """Build a deterministic high-contrast synthetic IDE/terminal screenshot PNG."""
    stride = width * 3
    buf = bytearray([24, 28, 36] * (width * height))

    def fill_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        cx0, cx1 = max(0, x0), min(width, x1)
        if cx1 <= cx0:
            return
        row_bytes = bytes(color) * (cx1 - cx0)
        for y in range(max(0, y0), min(height, y1)):
            start = y * stride + cx0 * 3
            buf[start : start + len(row_bytes)] = row_bytes

    def draw_text(x: int, y: int, text: str, scale: int, color: tuple[int, int, int]) -> None:
        cursor = x
        for ch in text:
            glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
            for gy, row_str in enumerate(glyph):
                for gx, bit in enumerate(row_str):
                    if bit == "1":
                        fill_rect(
                            cursor + gx * scale,
                            y + gy * scale,
                            cursor + (gx + 1) * scale,
                            y + (gy + 1) * scale,
                            color,
                        )
            cursor += 6 * scale

    # Title bar, sidebar, editor pane, and terminal panel with readable banner text.
    fill_rect(0, 0, width, max(4, height // 14), (40, 46, 58))
    fill_rect(
        width // 48,
        height // 64,
        width // 48 + max(4, width // 80),
        height // 64 + max(4, height // 80),
        (225, 85, 85),
    )
    fill_rect(
        width // 22,
        height // 64,
        width // 22 + max(4, width // 80),
        height // 64 + max(4, height // 80),
        (235, 185, 70),
    )
    fill_rect(
        width // 14,
        height // 64,
        width // 14 + max(4, width // 80),
        height // 64 + max(4, height // 80),
        (95, 200, 115),
    )
    fill_rect(0, max(4, height // 14), width // 5, height, (31, 36, 46))
    # Editor syntax bars
    for row in range(6):
        ry = int(height * 0.14) + row * max(8, height // 16)
        fill_rect(
            width // 4,
            ry,
            min(width - 24, width // 4 + 140 + row * 40),
            ry + max(4, height // 48),
            (110, 175, 245),
        )
    # Bottom terminal panel
    term_y = int(height * 0.58)
    fill_rect(width // 5, term_y, width, height, (16, 18, 24))
    fill_rect(width // 5, term_y, width, term_y + 3, (78, 150, 235))
    scale = max(1, min(width // 128, height // 96))
    draw_text(width // 4, term_y + max(8, height // 24), "STATUS: OK 402", scale, (115, 230, 140))
    draw_text(width // 4, term_y + max(24, height // 8), "AYAKA", scale, (240, 244, 250))
    return encode_png_rgb(width, height, bytes(buf))


def decode_base64_image(
    data_b64: str, *, media_type: str = "image/png", where: str = "image"
) -> ImageInput:
    """Decode raw base64 image payload (used by Anthropic Messages ``base64`` blocks)."""
    if not isinstance(data_b64, str) or not data_b64:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise InvalidRequestError(
            f"{where}: base64 data must be non-empty", context={"param": where}
        )
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            f"{where}: invalid base64 image payload: {exc}", context={"param": where}
        ) from exc
    return ImageInput(data=raw, media_type=media_type)


def decode_data_url_image(url: str, *, where: str = "image_url") -> ImageInput:
    """Decode a ``data:image/...;base64,...`` URL, refusing remote HTTP/HTTPS fetches."""
    if not isinstance(url, str) or not url:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise InvalidRequestError(
            f"{where}: must be a non-empty image URL", context={"param": where}
        )
    lower = url.lower()
    if lower.startswith(("http://", "https://", "ftp://", "file://")):
        raise InvalidRequestError(
            f"{where}: remote or file image URLs are refused to prevent silent outbound "
            "fetches; send base64 data:image/png;base64,... payloads instead",
            context={"param": where},
        )
    if not lower.startswith("data:"):
        raise InvalidRequestError(
            f"{where}: expected a data:image/png;base64,... URI", context={"param": where}
        )
    header, sep, payload = url[5:].partition(",")
    if not sep:
        raise InvalidRequestError(f"{where}: malformed data URI", context={"param": where})
    parts = [part.strip().lower() for part in header.split(";") if part.strip()]
    media_type = parts[0] if parts and "/" in parts[0] else ""
    if "base64" not in parts[1:]:
        raise InvalidRequestError(
            f"{where}: data URI must be base64-encoded", context={"param": where}
        )
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            f"{where}: invalid base64 image payload: {exc}", context={"param": where}
        ) from exc
    return ImageInput(data=raw, media_type=media_type or "image/png")
