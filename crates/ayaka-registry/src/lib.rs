//! Global model architecture registry.
//!
//! Maps a canonical `arch_id` (e.g. `"qwen3"`) to a [`ModelLoaderFactory`].
//! Concrete model crates register their factory at startup; the loader looks
//! one up by the architecture parsed from `config.json`/GGUF metadata.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use ayaka_loader::ModelLoaderFactory;

type Registry = RwLock<HashMap<&'static str, Arc<dyn ModelLoaderFactory>>>;

static MODEL_REGISTRY: OnceLock<Registry> = OnceLock::new();

fn registry() -> &'static Registry {
    MODEL_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a factory under its `arch_id`. A later registration with the same
/// id replaces the earlier one.
pub fn register(factory: Arc<dyn ModelLoaderFactory>) {
    let id = factory.arch_id();
    registry()
        .write()
        .expect("model registry poisoned")
        .insert(id, factory);
}

/// Look up a factory by architecture id.
pub fn get(arch_id: &str) -> Option<Arc<dyn ModelLoaderFactory>> {
    registry()
        .read()
        .expect("model registry poisoned")
        .get(arch_id)
        .cloned()
}

/// Architecture ids currently registered.
pub fn registered_archs() -> Vec<&'static str> {
    registry()
        .read()
        .expect("model registry poisoned")
        .keys()
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ayaka_loader::{DeviceMap, LoadedWeights, NormalModel, Result};

    struct DummyFactory;

    impl ModelLoaderFactory for DummyFactory {
        fn arch_id(&self) -> &'static str {
            "dummy-arch"
        }
        fn load(
            &self,
            _weights: LoadedWeights,
            _device_map: &DeviceMap,
        ) -> Result<Box<dyn NormalModel>> {
            unimplemented!("not exercised in registry tests")
        }
    }

    fn register_then_lookup() {
        register(Arc::new(DummyFactory));
        let f = get("dummy-arch").expect("registered factory should be found");
        assert_eq!(f.arch_id(), "dummy-arch");
        assert!(get("nonexistent").is_none());
        assert!(registered_archs().contains(&"dummy-arch"));
    }
}
