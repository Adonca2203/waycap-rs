fn main() {
    // CUDA FFI bindings
		#[cfg(feature = "nvidia")]
    println!("cargo:rustc-link-lib=dylib=cuda");

    println!("cargo:rustc-link-search=native=/usr/lib");
}
