//! Custom GGUF binary parser (no external GGUF dependency).
//!
//! Parses the header, metadata key/value store, and tensor table, then loads
//! tensors via one of two paths:
//!   * `load()` — dequantizes every tensor to F16/BF16 (kernels never see quant
//!     layouts);
//!   * `load_quantized()` — keeps quantized weights packed as `QTensor` so
//!     kernels CAN see quant layouts and dequantize on-the-fly (Phase 1.1).
//!
//! GGUF stores tensor dims fastest-varying first; we reverse them so the
//! resulting Candle tensors match Hugging Face `[out, in]` row-major shapes.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use ayaka_quant::{GgufDtype, QTensor};

use crate::error::{LoaderError, Result};
use crate::metadata::ModelMetadata;
use crate::weights::{LoadedQuantWeights, LoadedWeights};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32;

/// A scalar metadata value. Arrays are summarized by element type and length
/// (contents are skipped — only lengths, e.g. vocab size, are needed).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array { elem_type: u32, len: usize },
}

impl GgufValue {
    /// Coerce any integer value to `usize`.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            GgufValue::U8(v) => Some(*v as usize),
            GgufValue::I8(v) => Some(*v as usize),
            GgufValue::U16(v) => Some(*v as usize),
            GgufValue::I16(v) => Some(*v as usize),
            GgufValue::U32(v) => Some(*v as usize),
            GgufValue::I32(v) => Some(*v as usize),
            GgufValue::U64(v) => Some(*v as usize),
            GgufValue::I64(v) => Some(*v as usize),
            GgufValue::Array { len, .. } => Some(*len),
            _ => None,
        }
    }

    /// Coerce a float value to `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor's table entry.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Dimensions in Hugging Face order (`[out, in, ...]`), already reversed.
    pub dims: Vec<usize>,
    pub dtype: GgufDtype,
    /// Byte offset of this tensor's data, relative to the data section start.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total scalar element count.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// On-disk byte size of this tensor's stored blocks.
    pub fn stored_bytes(&self) -> usize {
        let block = self.dtype.block();
        let n_blocks = self
            .num_elements()
            .div_ceil(block.weights_per_block);
        n_blocks * block.bytes_per_block
    }
}

/// Parsed GGUF structure (everything except the raw tensor data).
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Absolute file offset where the tensor data section begins.
    pub data_offset: usize,
}

impl GgufFile {
    /// Parse the header, metadata, and tensor table from `buf`.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        let mut r = Reader::new(buf);
        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            return Err(LoaderError::Gguf(format!("bad magic {magic:#x}")));
        }
        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(LoaderError::Gguf(format!("unsupported version {version}")));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.string()?;
            let value = r.value()?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()? as usize);
            }
            dims.reverse(); // GGUF is fastest-varying-first; HF is [out, in].
            let raw_dtype = r.u32()?;
            let dtype = GgufDtype::from_ggml_id(raw_dtype)
                .ok_or_else(|| LoaderError::Gguf(format!("unknown ggml dtype {raw_dtype}")))?;
            let offset = r.u64()?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_usize)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let data_offset = r.pos().next_multiple_of(alignment);

        Ok(Self {
            metadata,
            tensors,
            data_offset,
        })
    }

    /// `general.architecture`.
    pub fn architecture(&self) -> Result<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .ok_or_else(|| LoaderError::Gguf("missing general.architecture".to_string()))
    }

    /// Build [`ModelMetadata`] from the GGUF metadata KV store.
    pub fn model_metadata(&self) -> Result<ModelMetadata> {
        let arch = self.architecture()?.to_string();
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let usize_at = |suffix: &str| {
            self.metadata
                .get(&key(suffix))
                .and_then(GgufValue::as_usize)
        };
        let req = |suffix: &str| -> Result<usize> {
            usize_at(suffix).ok_or_else(|| LoaderError::Gguf(format!("missing {}", key(suffix))))
        };

        let vocab_size = usize_at("vocab_size")
            .or_else(|| {
                self.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(GgufValue::as_usize)
            })
            .ok_or_else(|| LoaderError::Gguf("cannot determine vocab_size".to_string()))?;

        Ok(ModelMetadata {
            arch_id: arch.clone(),
            hidden_size: req("embedding_length")?,
            intermediate_size: req("feed_forward_length")?,
            num_hidden_layers: req("block_count")?,
            num_attention_heads: req("attention.head_count")?,
            num_key_value_heads: usize_at("attention.head_count_kv"),
            head_dim: usize_at("attention.key_length"),
            vocab_size,
            rms_norm_eps: self
                .metadata
                .get(&key("attention.layer_norm_rms_epsilon"))
                .and_then(GgufValue::as_f32)
                .map(|v| v as f64)
                .unwrap_or(1e-6),
            rope_theta: self
                .metadata
                .get(&key("rope.freq_base"))
                .and_then(GgufValue::as_f32)
                .unwrap_or(10_000.0),
            max_position_embeddings: usize_at("context_length").unwrap_or(4096),
            tie_word_embeddings: false,
            mla: None,
            moe: None,
        })
    }
}

/// Load a `.gguf` file into [`LoadedWeights`], dequantizing every tensor to `dtype`.
pub fn load(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<LoadedWeights> {
    let file = fs::File::open(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    // SAFETY: read-only mmap; file is not mutated while mapped.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let gguf = GgufFile::parse(&mmap)?;
    let metadata = gguf.model_metadata()?;

    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(gguf.tensors.len());
    let mut weight_bytes = 0usize;
    let elem = dtype.size_in_bytes();

    for info in &gguf.tensors {
        let start = gguf.data_offset + info.offset as usize;
        let end = start + info.stored_bytes();
        if end > mmap.len() {
            return Err(LoaderError::Gguf(format!(
                "tensor {} runs past end of file",
                info.name
            )));
        }
        let raw = &mmap[start..end];
        let f32_data = info.dtype.dequantize(raw, info.num_elements())?;
        let tensor = Tensor::from_vec(f32_data, info.dims.clone(), device)?.to_dtype(dtype)?;
        weight_bytes += info.num_elements() * elem;
        tensors.insert(info.name.clone(), tensor);
    }

    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    Ok(LoadedWeights::new(metadata, vb, weight_bytes))
}

/// Owns a GGUF file's mmap and parsed header for safe zero-copy tensor slicing.
///
/// The `Mmap` stays alive for the lifetime of this struct, so `tensor_slice()`
/// returns valid borrows of the file's data section. When loading quantized
/// weights, we copy each tensor's packed bytes into a `QTensor` (owning),
/// so the `GgufMmap` can be dropped after `load_quantized()` returns.
pub struct GgufMmap {
    _file: fs::File,
    mmap: memmap2::Mmap,
    gguf: GgufFile,
}

impl GgufMmap {
    /// Open and mmap a `.gguf` file (read-only), parsing the header.
    pub fn open(path: &Path) -> Result<Self> {
        let file = fs::File::open(path).map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // SAFETY: read-only mmap; file is not mutated while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let gguf = GgufFile::parse(&mmap)?;
        Ok(Self {
            _file: file,
            mmap,
            gguf,
        })
    }

    /// Borrow the parsed GGUF header/metadata/tensor table.
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// Borrow the raw bytes of tensor `info` from the mmap'd data section.
    /// Returns `None` if the tensor runs past the end of the file.
    pub fn tensor_slice(
        &self,
        info: &GgufTensorInfo,
    ) -> Option<&[u8]> {
        let start = self.gguf.data_offset + info.offset as usize;
        let end = start + info.stored_bytes();
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }
}

/// Load a `.gguf` file keeping quantized weights in their packed block format.
///
/// Quantized tensors (Q4K, Q6K, Q4_0, Q8_0) are stored as `QTensor` (raw packed
/// bytes + metadata) without dequantizing. Non-quantized tensors (F16, BF16, F32)
/// are materialized to `dtype` on `device` via a `VarBuilder`, exactly as `load()`
/// does — they're small (embeddings, norms) and kernels consume them directly.
///
/// `weight_bytes` reports the **quantized** on-disk byte count (not the
/// dequantized size), matching what actually resides in VRAM.
pub fn load_quantized(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<LoadedQuantWeights> {
    let mmap = GgufMmap::open(path)?;
    let gguf = mmap.gguf();
    let metadata = gguf.model_metadata()?;

    let mut qtensors: HashMap<String, QTensor> = HashMap::new();
    let mut float_tensors: HashMap<String, Tensor> = HashMap::new();
    let mut weight_bytes = 0usize;

    for info in &gguf.tensors {
        let raw = mmap.tensor_slice(info).ok_or_else(|| {
            LoaderError::Gguf(format!("tensor {} runs past end of file", info.name))
        })?;

        let scheme = info.dtype.scheme();
        if scheme.is_float() {
            // F16/BF16/F32: materialize to `dtype` on `device` (same as legacy load).
            let f32_data = info.dtype.dequantize(raw, info.num_elements())?;
            let tensor = Tensor::from_vec(f32_data, info.dims.clone(), device)?.to_dtype(dtype)?;
            weight_bytes += info.num_elements() * dtype.size_in_bytes();
            float_tensors.insert(info.name.clone(), tensor);
        } else {
            // Quantized: keep packed bytes as QTensor (no dequant, no GPU upload yet).
            let qt = QTensor::from_raw(raw.to_vec(), info.dims.clone(), scheme)?;
            weight_bytes += qt.stored_bytes();
            qtensors.insert(info.name.clone(), qt);
        }
    }

    let vb = VarBuilder::from_tensors(float_tensors, dtype, device);
    Ok(LoadedQuantWeights::new(
        metadata,
        vb,
        qtensors,
        weight_bytes,
    ))
}

/// Little-endian cursor over GGUF bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(
        &mut self,
        n: usize,
    ) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| LoaderError::Gguf("length overflow".to_string()))?;
        if end > self.buf.len() {
            return Err(LoaderError::Gguf("unexpected end of file".to_string()));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read a metadata value given its leading type tag.
    fn value(&mut self) -> Result<GgufValue> {
        let ty = self.u32()?;
        self.value_of_type(ty)
    }

    fn value_of_type(
        &mut self,
        ty: u32,
    ) -> Result<GgufValue> {
        Ok(match ty {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.u8()? as i8),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.u16()? as i16),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(self.f32()?),
            7 => GgufValue::Bool(self.u8()? != 0),
            8 => GgufValue::String(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let len = self.u64()? as usize;
                for _ in 0..len {
                    // Advance past each element; contents are discarded.
                    self.value_of_type(elem_type)?;
                }
                GgufValue::Array { elem_type, len }
            },
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(self.f64()?),
            other => return Err(LoaderError::Gguf(format!("unknown value type {other}"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ayaka_quant::QuantScheme;

    /// Hand-build a minimal v3 GGUF with metadata and one F32 tensor.
    fn build_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&2u64.to_le_bytes()); // kv_count

        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        // KV 1: general.architecture = "llama" (type 8 = string)
        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut b, "llama");
        // KV 2: llama.block_count = 12 (type 4 = u32)
        push_str(&mut b, "llama.block_count");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&12u32.to_le_bytes());

        // Tensor info: name="w", ndims=2, dims=[8,4] (ggml order), F32, offset 0.
        push_str(&mut b, "w");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes()); // ne[0] = 8 (fastest)
        b.extend_from_slice(&4u64.to_le_bytes()); // ne[1] = 4
        b.extend_from_slice(&0u32.to_le_bytes()); // ggml type 0 = F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Pad to 32-byte alignment, then 32 f32 values.
        while b.len() % DEFAULT_ALIGNMENT != 0 {
            b.push(0);
        }
        for i in 0..32u32 {
            b.extend_from_slice(&(i as f32).to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_header_metadata_and_tensor_table() {
        let buf = build_gguf();
        let gguf = GgufFile::parse(&buf).unwrap();
        assert_eq!(gguf.architecture().unwrap(), "llama");
        assert_eq!(
            gguf.metadata
                .get("llama.block_count")
                .unwrap()
                .as_usize(),
            Some(12)
        );
        assert_eq!(gguf.tensors.len(), 1);
        let t = &gguf.tensors[0];
        assert_eq!(t.name, "w");
        // dims reversed from ggml [8,4] -> HF [4,8].
        assert_eq!(t.dims, vec![4, 8]);
        assert_eq!(t.dtype, GgufDtype::F32);
        assert_eq!(gguf.data_offset % DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn loads_tensor_with_reversed_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        fs::write(&path, build_gguf()).unwrap();

        // block_count present but other required fields missing -> metadata
        // inference fails; verify the raw tensor load path via GgufFile instead.
        let buf = fs::read(&path).unwrap();
        let gguf = GgufFile::parse(&buf).unwrap();
        let info = &gguf.tensors[0];
        let start = gguf.data_offset + info.offset as usize;
        let raw = &buf[start..start + info.stored_bytes()];
        let data = info
            .dtype
            .dequantize(raw, info.num_elements())
            .unwrap();
        assert_eq!(data.len(), 32);
        assert_eq!(data[5], 5.0);

        let tensor = Tensor::from_vec(data, info.dims.clone(), &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[4, 8]);
    }

    /// Spec for building a test tensor entry in `build_complete_gguf`.
    struct TensorSpec {
        name: &'static str,
        /// `(ne[0], ne[1])` in ggml fastest-varying-first order.
        ne: [u64; 2],
        /// ggml type id (0 = F32, 1 = F16, 12 = Q4K, ...).
        ggml_type: u32,
        /// Raw bytes to write for this tensor's data section.
        data: Vec<u8>,
    }

    /// Build a complete v3 GGUF with all metadata required by
    /// `GgufFile::model_metadata()` plus the given tensors. Tensors are laid out
    /// back-to-back, each aligned to `DEFAULT_ALIGNMENT`.
    fn build_complete_gguf(tensors: &[TensorSpec]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        b.extend_from_slice(&6u64.to_le_bytes()); // kv_count

        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        let push_kv_u32 = |b: &mut Vec<u8>, k: &str, v: u32| {
            push_str(b, k);
            b.extend_from_slice(&4u32.to_le_bytes()); // type 4 = u32
            b.extend_from_slice(&v.to_le_bytes());
        };
        let push_kv_str = |b: &mut Vec<u8>, k: &str, v: &str| {
            push_str(b, k);
            b.extend_from_slice(&8u32.to_le_bytes()); // type 8 = string
            push_str(b, v);
        };

        // Required metadata for `model_metadata()`.
        push_kv_str(&mut b, "general.architecture", "llama");
        push_kv_u32(&mut b, "llama.embedding_length", 8);
        push_kv_u32(&mut b, "llama.feed_forward_length", 16);
        push_kv_u32(&mut b, "llama.block_count", 1);
        push_kv_u32(&mut b, "llama.attention.head_count", 2);
        push_kv_u32(&mut b, "llama.vocab_size", 4);

        // Compute aligned offsets for each tensor relative to the data section.
        let mut offset = 0usize;
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            offsets.push(offset);
            offset = (offset + t.data.len()).next_multiple_of(DEFAULT_ALIGNMENT);
        }

        // Write tensor table entries.
        for (t, &off) in tensors.iter().zip(&offsets) {
            push_str(&mut b, t.name);
            b.extend_from_slice(&2u32.to_le_bytes()); // ndims = 2
            b.extend_from_slice(&t.ne[0].to_le_bytes());
            b.extend_from_slice(&t.ne[1].to_le_bytes());
            b.extend_from_slice(&t.ggml_type.to_le_bytes());
            b.extend_from_slice(&(off as u64).to_le_bytes());
        }

        // Pad to alignment for the data section.
        while b.len() % DEFAULT_ALIGNMENT != 0 {
            b.push(0);
        }
        let data_section_start = b.len();

        // Write tensor data with alignment padding between tensors.
        for (i, t) in tensors.iter().enumerate() {
            let expected = data_section_start + offsets[i];
            while b.len() < expected {
                b.push(0);
            }
            b.extend_from_slice(&t.data);
        }
        b
    }

    #[test]
    fn gguf_mmap_open_and_slice() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        fs::write(&path, build_gguf()).unwrap();

        let mmap = GgufMmap::open(&path).expect("open should parse header");
        let gguf = mmap.gguf();
        assert_eq!(gguf.architecture().unwrap(), "llama");
        assert_eq!(gguf.tensors.len(), 1);

        let info = &gguf.tensors[0];
        let raw = mmap
            .tensor_slice(info)
            .expect("tensor slice should exist");
        assert_eq!(raw.len(), info.stored_bytes());
        // The first 4 bytes are the first f32 value (0.0).
        let first = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        assert_eq!(first, 0.0);
        // The 6th value (index 5) should be 5.0.
        let sixth = f32::from_le_bytes([raw[20], raw[21], raw[22], raw[23]]);
        assert_eq!(sixth, 5.0);
    }

    #[test]
    fn load_quantized_keeps_packed_bytes() {
        // F32 tensor "embed": dims [4, 8] HF = [8, 4] ggml, 32 floats.
        let embed_data: Vec<u8> = (0..32u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        // Q4K tensor "layer.q_proj.weight": dims [1, 256] HF = [256, 1] ggml,
        // 1 super-block = 144 bytes (zeros — test only checks preservation).
        let q4k_data = vec![0u8; 144];

        let buf = build_complete_gguf(&[
            TensorSpec {
                name: "embed",
                ne: [8, 4],
                ggml_type: 0, // F32
                data: embed_data,
            },
            TensorSpec {
                name: "layer.q_proj.weight",
                ne: [256, 1],
                ggml_type: 12, // Q4K
                data: q4k_data,
            },
        ]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.gguf");
        fs::write(&path, &buf).unwrap();

        let result =
            load_quantized(&path, DType::F32, &Device::Cpu).expect("load_quantized should succeed");

        // Q4K tensor stays packed as QTensor.
        let qt = result
            .qtensors
            .get("layer.q_proj.weight")
            .expect("Q4K tensor should be in qtensors");
        assert_eq!(qt.scheme, QuantScheme::Q4K);
        assert_eq!(qt.stored_bytes(), 144);
        assert_eq!(qt.dims, vec![1, 256]);

        // F32 tensor is materialized into the VarBuilder.
        let embed = result
            .vb
            .get((4, 8), "embed")
            .expect("embed should be in VarBuilder");
        assert_eq!(embed.dims(), &[4, 8]);
        assert_eq!(embed.dtype(), DType::F32);

        // weight_bytes = 144 (Q4K packed) + 128 (32 * 4 bytes F32) = 272.
        assert_eq!(result.weight_bytes, 272);
    }

    #[test]
    fn load_quantized_mixed_dtypes() {
        // F16 tensor "norm": dims [4, 8] HF = [8, 4] ggml, 32 f16 values.
        let norm_data: Vec<u8> = (0..32u16)
            .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
            .collect();
        // Q4K tensor "layer.q_proj.weight": dims [1, 256], 144 bytes.
        let q4k_data = vec![0u8; 144];

        let buf = build_complete_gguf(&[
            TensorSpec {
                name: "norm",
                ne: [8, 4],
                ggml_type: 1, // F16
                data: norm_data,
            },
            TensorSpec {
                name: "layer.q_proj.weight",
                ne: [256, 1],
                ggml_type: 12, // Q4K
                data: q4k_data,
            },
        ]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f16_q4k.gguf");
        fs::write(&path, &buf).unwrap();

        let result =
            load_quantized(&path, DType::F32, &Device::Cpu).expect("load_quantized should succeed");

        // F16 tensor goes to VarBuilder (materialized to F32).
        let norm = result
            .vb
            .get((4, 8), "norm")
            .expect("F16 tensor should be in VarBuilder");
        assert_eq!(norm.dims(), &[4, 8]);

        // Q4K tensor goes to qtensors.
        assert!(
            result
                .qtensors
                .contains_key("layer.q_proj.weight")
        );
        assert!(!result.qtensors.contains_key("norm"));
    }
}
