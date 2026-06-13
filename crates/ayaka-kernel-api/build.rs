use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_dir = manifest_dir.join("..").join("..").join("native");

    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("include").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("csrc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("CMakeLists.txt").display()
    );
    println!("cargo:rerun-if-env-changed=AYAKA_CUDA_ARCHS");

    let mut cfg = cmake::Config::new(&native_dir);
    cfg.define("AYAKA_ENABLE_CUDA", "ON")
        .profile("Release");
    if let Ok(archs) = env::var("AYAKA_CUDA_ARCHS") {
        cfg.define("CMAKE_CUDA_ARCHITECTURES", archs);
    }
    let dst = cfg.build_target("ayaka-native").build();

    // Single-config generators put the lib in build/; multi-config (MSVC) in
    // build/<Config>/.
    for sub in ["build", "build/Release", "build/Debug"] {
        println!("cargo:rustc-link-search=native={}", dst.join(sub).display());
    }
    println!("cargo:rustc-link-lib=static=ayaka-native");

    link_cuda_runtime();
}

fn link_cuda_runtime() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!(
                "cargo:rustc-link-search=native={}",
                PathBuf::from(cuda_path)
                    .join("lib")
                    .join("x64")
                    .display()
            );
        }
        println!("cargo:rustc-link-lib=static=cudart_static");
        // Import library; resolves to cublasLt64_*.dll at runtime.
        println!("cargo:rustc-link-lib=cublasLt");
    } else {
        for dir in ["/usr/local/cuda/lib64", "/opt/cuda/lib64"] {
            if PathBuf::from(dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
            }
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
