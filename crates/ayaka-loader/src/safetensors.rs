//! Safetensors loading: mmap one or more shards into a [`VarBuilder`].

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::error::{LoaderError, Result};
use crate::metadata::ModelMetadata;
use crate::weights::LoadedWeights;

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
}
