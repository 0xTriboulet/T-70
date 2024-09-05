use std::fs;

fn main() {
    let mut build = cc::Build::new();

    // Specify the Clang compiler explicitly.
    build.compiler("clang++");

    // Set the standard to C++20
    build.flag_if_supported("-std=c++20");

    // Disable linking the C++ standard library (nostdlib++)
    build.flag_if_supported("-nostdlib++");

    // Optionally disable warnings for missing field initializers
    build.flag_if_supported("-Wno-missing-field-initializers");

    // Set optimization level based on profile
    let is_release = std::env::var("PROFILE").unwrap() == "release";
    if is_release {
        // Disable all warnings in release mode
        build.flag_if_supported("-w");
        build.opt_level(3); // Higher optimization for release builds
    } else {
        build.opt_level(2); // Set lower optimization for non-release builds
    }

    // Specify the directory that contains the C++ source files.
    let c_src_path = "c_src";

    // Collect all .c or .cxx files in the c_src directory and add them to the build.
    let c_files = fs::read_dir(c_src_path)
        .expect("Failed to read c_src directory")
        .filter_map(|entry| {
            let path = entry.expect("Invalid entry").path();
            if path.extension().and_then(|s| s.to_str()) == Some("c") || path.extension().and_then(|s| s.to_str()) == Some("cxx") {
                Some(path)
            } else {
                None
            }
        });

    for file in c_files {
        println!("Compiling C/C++ file: {:?}", file);
        build.file(file);
    }

    // Compile the C++ code as a static library.
    build.compile("cpp_lib");

    // Instruct Cargo to re-run build.rs if any of the files in c_src change.
    println!("cargo:rerun-if-changed={}", c_src_path);
    println!("cargo:rustc-link-lib=static=cpp_lib"); // Link to static library
}
