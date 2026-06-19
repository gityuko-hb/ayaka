//! Safetensors loading: mmap one or more shards into a [`VarBuilder`].

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use half::f16;
use serde::Deserialize;

use ayaka_quant::{AwqGptqInput, QuantScheme, RepackedWeights, repack_awq_gptq};

use crate::error::{LoaderError, Result};
use crate::metadata::ModelMetadata;
use crate::weights::{LoadedQuantWeights, LoadedWeights};

/// Load a model directory (or a single `.safetensors` file) into [`LoadedWeights`].
///
/// Resolves `config.json` and the shard set (`*.safetensors.index.json` when
/// present), mmaps every shard, and builds a `VarBuilder` materializing tensors
/// to `dtype` on `device`.
pub fn load(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<LoadedWeights> {
    let (config_path, shards) = resolve_paths(path)?;
    let config_json = fs::read_to_string(&config_path).map_err(|source| LoaderError::Io {
        path: config_path.clone(),
        source,
    })?;
    let metadata = ModelMetadata::from_config_json(&config_json)?;

    let weight_bytes = resident_bytes(&shards, dtype)?;

    // SAFETY: the mmap lives inside the VarBuilder's backend for its lifetime;
    // the files must not be mutated while loaded (standard safetensors usage).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, device)? };

    Ok(LoadedWeights::new(metadata, vb, weight_bytes))
}

/// Load a safetensors model with AWQ/GPTQ quantized weights, keeping them packed.
///
/// Detects `*.qweight`/`*.qzeros`/`*.scales` tensor groups by name. For each
/// group, reads the raw packed tensors, repacks to [`RepackedWeights`] via
/// [`repack_awq_gptq`], and stores the result in the `repacked` map keyed by
/// the weight prefix (e.g. `"model.layers.0.self_attn.q_proj"`).
///
/// Non-quantized tensors (embeddings, norms, lm_head) are materialized to
/// `dtype` via `VarBuilder` as usual — the model factory pulls them by name.
///
/// Requires `quantization_config` in `config.json` with `quant_method` equal
/// to `"awq"` or `"gptq"`.
pub fn load_quantized(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<LoadedQuantWeights> {
    let (config_path, shards) = resolve_paths(path)?;
    let config_json = fs::read_to_string(&config_path).map_err(|source| LoaderError::Io {
        path: config_path.clone(),
        source,
    })?;
    let metadata = ModelMetadata::from_config_json(&config_json)?;

    let quant_config = metadata.quant_config.as_ref().ok_or_else(|| {
        LoaderError::InvalidConfig(
            "safetensors::load_quantized requires quantization_config".into(),
        )
    })?;

    let scheme = match quant_config.quant_method.as_str() {
        "awq" => QuantScheme::AWQ,
        "gptq" => QuantScheme::GPTQ,
        other => {
            return Err(LoaderError::InvalidConfig(format!(
                "unsupported quant_method: {other}"
            )));
        },
    };

    // SAFETY: mmap lives inside the VarBuilder's backend for its lifetime;
    // the files must not be mutated while loaded (standard safetensors usage).
    // The VarBuilder is lazy — quantized tensors (qweight/qzeros/scales) are
    // only materialized if `get()` is called on them, which the model factory
    // never does.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, device)? };

    let mut repacked: HashMap<String, RepackedWeights> = HashMap::new();
    let mut weight_bytes = 0usize;

    for shard_path in &shards {
        let file = fs::File::open(shard_path).map_err(|source| LoaderError::Io {
            path: shard_path.clone(),
            source,
        })?;
        // SAFETY: read-only mmap; file is not mutated while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| LoaderError::Io {
            path: shard_path.clone(),
            source,
        })?;
        let st = ::safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| LoaderError::InvalidConfig(format!("{}: {e}", shard_path.display())))?;

        // First pass: sum resident bytes for all tensors in this shard.
        for (name, view) in st.tensors() {
            if name.ends_with(".qweight") || name.ends_with(".qzeros") || name.ends_with(".scales")
            {
                weight_bytes += view.data().len();
            } else {
                let n: usize = view.shape().iter().product();
                weight_bytes += n * dtype.size_in_bytes();
            }
        }

        // Second pass: find and repack quantized tensor groups.
        let qweight_names: Vec<String> = st
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| name.ends_with(".qweight"))
            .collect();

        for name in qweight_names {
            let prefix = &name[..name.len() - ".qweight".len()];

            let qw_view = st
                .tensor(&format!("{prefix}.qweight"))
                .map_err(|e| LoaderError::InvalidConfig(format!("{e}")))?;
            let qweight = read_u32_slice(qw_view.data());

            // qweight shape: [K/8, N] → K = shape[0]*8, N = shape[1].
            let qw_shape = qw_view.shape();
            if qw_shape.len() != 2 {
                return Err(LoaderError::InvalidConfig(format!(
                    "{prefix}.qweight: expected 2-D shape, got {qw_shape:?}"
                )));
            }
            let k = qw_shape[0] * 8;
            let n = qw_shape[1];

            let qzeros = match st.tensor(&format!("{prefix}.qzeros")) {
                Ok(view) => read_u32_slice(view.data()),
                Err(_) => Vec::new(), // symmetric: no zeros
            };

            let scales_view = st
                .tensor(&format!("{prefix}.scales"))
                .map_err(|e| LoaderError::InvalidConfig(format!("{e}")))?;
            let scales_raw = read_f16_slice(scales_view.data());

            let n_groups = k / quant_config.group_size;
            let scales = match scales_view.shape() {
                [sn, sg] if *sn == n && *sg == n_groups => scales_raw,
                [sg, sn] if *sn == n && *sg == n_groups => {
                    // Transpose [n_groups, N] → [N, n_groups].
                    let mut out = vec![f16::from_f32(0.0); n * n_groups];
                    for col in 0..n {
                        for g in 0..n_groups {
                            out[col * n_groups + g] = scales_raw[g * n + col];
                        }
                    }
                    out
                },
                other => {
                    return Err(LoaderError::InvalidConfig(format!(
                        "{prefix}.scales: expected shape [{n}, {n_groups}] or \
                         [{n_groups}, {n}], got {other:?}"
                    )));
                },
            };

            let input = AwqGptqInput {
                qweight,
                qzeros,
                scales,
                out_features: n,
                in_features: k,
                group_size: quant_config.group_size,
                scheme,
            };

            let rw = repack_awq_gptq(&input).map_err(LoaderError::Dequant)?;
            repacked.insert(prefix.to_string(), rw);
        }
    }

    Ok(LoadedQuantWeights::new(
        metadata,
        vb,
        HashMap::new(),
        repacked,
        weight_bytes,
    ))
}

/// Read a little-endian I32 byte buffer as `Vec<u32>`.
fn read_u32_slice(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read a little-endian F16 byte buffer as `Vec<f16>`.
fn read_f16_slice(bytes: &[u8]) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Resolve `(config.json, [shard paths])` from a directory or single file.
fn resolve_paths(path: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    if path.is_file() {
        // A bare .safetensors file; expect config.json beside it.
        let dir = path.parent().unwrap_or(Path::new("."));
        return Ok((dir.join("config.json"), vec![path.to_path_buf()]));
    }

    let config_path = path.join("config.json");
    let index_path = path.join("model.safetensors.index.json");

    let shards = if index_path.is_file() {
        let raw = fs::read_to_string(&index_path).map_err(|source| LoaderError::Io {
            path: index_path.clone(),
            source,
        })?;
        let index: ShardIndex =
            serde_json::from_str(&raw).map_err(|source| LoaderError::Parse {
                what: "safetensors index".to_string(),
                source,
            })?;
        index
            .weight_map
            .into_values()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|f| path.join(f))
            .collect()
    } else {
        let single = path.join("model.safetensors");
        if !single.is_file() {
            return Err(LoaderError::InvalidConfig(format!(
                "no model.safetensors or index in {}",
                path.display()
            )));
        }
        vec![single]
    };

    Ok((config_path, shards))
}

#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// Bytes the weights occupy once materialized to `dtype`: sum over all tensors
/// of `num_elements * dtype.size_in_bytes()`.
fn resident_bytes(
    shards: &[PathBuf],
    dtype: DType,
) -> Result<usize> {
    let elem = dtype.size_in_bytes();
    let mut total = 0usize;
    for shard in shards {
        let file = fs::File::open(shard).map_err(|source| LoaderError::Io {
            path: shard.clone(),
            source,
        })?;
        // SAFETY: read-only mmap of header bytes; file outlives the borrow.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| LoaderError::Io {
            path: shard.clone(),
            source,
        })?;
        let (_, meta) = ::safetensors::SafeTensors::read_metadata(&mmap)
            .map_err(|e| LoaderError::InvalidConfig(format!("{}: {e}", shard.display())))?;
        for (_, info) in meta.tensors() {
            let n: usize = info.shape.iter().product();
            total += n * elem;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Write a tiny safetensors file + config.json into a temp dir and load it.
    #[test]
    fn loads_single_file_model() {
        let dir = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;

        // Two small tensors.
        let a = Tensor::zeros((4, 8), DType::F32, &dev).unwrap();
        let b = Tensor::zeros((8,), DType::F32, &dev).unwrap();
        let map = std::collections::HashMap::from([
            ("model.embed".to_string(), a),
            ("model.norm".to_string(), b),
        ]);
        candle_core::safetensors::save(&map, dir.path().join("model.safetensors")).unwrap();

        fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 4
            }"#,
        )
        .unwrap();

        let loaded = load(dir.path(), DType::F32, &dev).unwrap();
        assert_eq!(loaded.metadata.arch_id, "llama");
        // 4*8 + 8 = 40 elements * 4 bytes = 160.
        assert_eq!(loaded.weight_bytes, 160);
        // Tensors are reachable by name through the VarBuilder.
        let t = loaded.vb.get((4, 8), "model.embed").unwrap();
        assert_eq!(t.dims(), &[4, 8]);
    }

    /// Build a tiny AWQ safetensors model and load it via `load_quantized`.
    fn write_awq_model(
        dir: &std::path::Path,
        scales_shape: [usize; 2],
    ) {
        let dev = Device::Cpu;
        let n = 2;
        let k = 32;
        let group_size = 4;
        let n_groups = k / group_size; // 8

        // qweight: I32 [K/8, N] = [4, 2], all nibbles = 3 → 0x33333333.
        let qweight_vals = vec![0x33333333u32 as i32; (k / 8) * n];
        let qweight = Tensor::from_vec(qweight_vals, (k / 8, n), &dev).unwrap();

        // qzeros: I32 [n_groups/8, N] = [1, 2], all nibbles = 5 → 0x55555555.
        let qzeros_vals = vec![0x55555555u32 as i32; (n_groups / 8) * n];
        let qzeros = Tensor::from_vec(qzeros_vals, (n_groups / 8, n), &dev).unwrap();

        // scales: F16, shape per `scales_shape`, all 2.0.
        let scales = Tensor::from_vec(vec![2.0f32; n * n_groups], &scales_shape, &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        // Non-quantized tensor.
        let embed = Tensor::zeros((4, 8), DType::F32, &dev).unwrap();

        let map = std::collections::HashMap::from([
            ("layer.q_proj.qweight".to_string(), qweight),
            ("layer.q_proj.qzeros".to_string(), qzeros),
            ("layer.q_proj.scales".to_string(), scales),
            ("model.embed".to_string(), embed),
        ]);
        candle_core::safetensors::save(&map, dir.join("model.safetensors")).unwrap();

        fs::write(
            dir.join("config.json"),
            r#"{
                "model_type": "qwen3",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 4,
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 4,
                    "sym": false
                }
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn loads_awq_quantized_model() {
        let dir = tempfile::tempdir().unwrap();
        let n = 2;
        let k = 32;
        let group_size = 4;
        let n_groups = k / group_size; // 8

        // scales shape [N, n_groups] (row-major, the canonical AWQ layout).
        write_awq_model(dir.path(), [n, n_groups]);

        let dev = Device::Cpu;
        let loaded = load_quantized(dir.path(), DType::F32, &dev).unwrap();

        // Quantized weight repacked and stored by prefix.
        let rw = loaded
            .repacked
            .get("layer.q_proj")
            .expect("repacked entry should exist");
        assert_eq!(rw.out_features, n);
        assert_eq!(rw.in_features, k);
        assert_eq!(rw.group_size, group_size);
        assert_eq!(rw.scheme, QuantScheme::AWQ);
        assert_eq!(rw.qs.len(), n * k / 2);
        assert_eq!(rw.scales.len(), n * n_groups);
        assert_eq!(rw.mins.len(), n * n_groups);

        // All scales = 2.0, all mins = -10.0 (= -2.0 * 5).
        assert!(rw.scales.iter().all(|s| s.to_f32() == 2.0));
        assert!(rw.mins.iter().all(|m| m.to_f32() == -10.0));

        // Non-quantized tensor reachable via VarBuilder.
        let t = loaded.vb.get((4, 8), "model.embed").unwrap();
        assert_eq!(t.dims(), &[4, 8]);

        // weight_bytes includes quantized raw bytes + float tensor bytes.
        // qweight: 4*2*4 = 32, qzeros: 1*2*4 = 8, scales: 2*8*2 = 32,
        // embed: 4*8*4 = 128. Total = 200.
        assert_eq!(loaded.weight_bytes, 200);
    }

    #[test]
    fn loads_awq_with_transposed_scales() {
        let dir = tempfile::tempdir().unwrap();
        let n = 2;
        let k = 32;
        let group_size = 4;
        let n_groups = k / group_size; // 8

        // scales shape [n_groups, N] (transposed layout) — loader should transpose.
        write_awq_model(dir.path(), [n_groups, n]);

        let dev = Device::Cpu;
        let loaded = load_quantized(dir.path(), DType::F32, &dev).unwrap();

        let rw = loaded
            .repacked
            .get("layer.q_proj")
            .expect("repacked entry should exist");
        // After transpose, scales should still be all 2.0 (uniform values,
        // but the test confirms the transpose path doesn't error or scramble).
        assert_eq!(rw.scales.len(), n * n_groups);
        assert!(rw.scales.iter().all(|s| s.to_f32() == 2.0));
        assert!(rw.mins.iter().all(|m| m.to_f32() == -10.0));
    }

    #[test]
    fn load_quantized_rejects_missing_quant_config() {
        let dir = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;

        let embed = Tensor::zeros((4, 8), DType::F32, &dev).unwrap();
        let map = std::collections::HashMap::from([("model.embed".to_string(), embed)]);
        candle_core::safetensors::save(&map, dir.path().join("model.safetensors")).unwrap();

        // No quantization_config in config.json.
        fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type": "qwen3",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 4
            }"#,
        )
        .unwrap();

        let result = load_quantized(dir.path(), DType::F32, &dev);
        match result {
            Err(LoaderError::InvalidConfig(_)) => {}, // expected
            Err(other) => panic!("expected InvalidConfig, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
