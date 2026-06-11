pub mod rmsnorm;

pub use rmsnorm::rmsnorm_ref;
#[cfg(feature = "cuda")]
pub use rmsnorm::{rmsnorm, rmsnorm_new};
