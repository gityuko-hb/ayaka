#pragma once

// Marlin tile geometry constants and device-side helpers shared by the
// W4A16 Marlin kernel family. The offline Rust shuffle in
// `crates/ayaka-quantization/src/repack.rs::repack_to_marlin` produces the
// `[N/16, K/16, 16, 8]` tile layout these constants describe.
//
// Tile inner layout: 16 rows x 8 bytes (16 nibbles per row). The Rust shuffle
// lays the 16 inner rows along the N (output-channel) axis and the 8 bytes
// along the K (input-feature) axis, matching `mma.sync.m16n8k16` operand
// expectations once Tensor Cores are wired in.

#include <cstdint>

namespace ayaka {

// Matches mma.sync.aligned.m16n8k16: M=16, N=8, K=16.
// N is tiled into groups of 16 (two mma N-tiles of 8 each).
constexpr int MARLIN_TILE_M = 16;
constexpr int MARLIN_TILE_N = 16;
constexpr int MARLIN_TILE_K = 16;

// 16 nibbles packed into 8 bytes per tile row.
constexpr int MARLIN_BYTES_PER_TILE_ROW = 8;

// Bytes in one 16x16 weight tile: 16 rows * 8 bytes.
constexpr int MARLIN_TILE_BYTES = MARLIN_TILE_N * MARLIN_BYTES_PER_TILE_ROW;

// Number of 4-bit weights per tile row.
constexpr int MARLIN_NIBBLES_PER_TILE_ROW = MARLIN_BYTES_PER_TILE_ROW * 2;

}  // namespace ayaka
