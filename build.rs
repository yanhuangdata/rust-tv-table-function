use anyhow::Context;
use std::{env, process::Command};
use zngur::Zngur;

fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=main.zng");
    println!("cargo:rerun-if-changed=ci/scripts/fix_generated.sh");
    println!("cargo:rerun-if-changed=zngur/extra_generated.h");

    let cwd = std::env::current_dir().context("failed to get current dir")?;
    let cpp_zngur = cwd.join("cpp").join("zngur");
    std::fs::create_dir_all(&cpp_zngur).context("Failed to create cpp zngur dir")?;
    let generated_header = cwd.join("cpp").join("zngur").join("generated.h");
    let extra_generated_header = cwd.join("cpp").join("zngur").join("extra_generated.h");
    let generated_cpp = cwd.join("cpp/zngur/generated.cpp");
    Zngur::from_zng_file(cwd.join("main.zng"))
        .with_cpp_file(&generated_cpp)
        .with_h_file(&generated_header)
        .with_rs_file(cwd.join("src/zngur_generated.rs"))
        .generate();

    let zngur_tvtf = cwd.join("cpp").join("src").join("zngur_tvtf");
    std::fs::create_dir_all(&zngur_tvtf).context("Failed to create zngur_tvtf dir")?;
    std::fs::copy(&generated_header, zngur_tvtf.join("generated.h"))
        .context("Failed to copy generated.h")?;
    std::fs::copy(
        &extra_generated_header,
        zngur_tvtf.join("extra_generated.h"),
    )
    .context("Failed to copy extra_generated.h")?;

    let script_path = "./ci/scripts/fix_generated.sh";
    let output = Command::new(script_path)
        .output()
        .context("failed to execute script")?;

    if std::env::var("CARGO_DEBUG").is_ok() {
        println!("cargo:warning=fix_generated.sh output:");
        println!("cargo:warning={}", String::from_utf8_lossy(&output.stdout));
        println!("cargo:warning={}", String::from_utf8_lossy(&output.stderr));
        println!("cargo:warning={}", &output.status);
    }

    let cxx = env::var("CXX").unwrap_or("c++".to_owned());
    let my_build = &mut cc::Build::new();
    let my_build = my_build
        .cpp(true)
        .std("c++17")
        .warnings(false) // auto generated
        .compiler(&cxx)
        .include(&cpp_zngur);
    let my_build = || my_build.clone();
    my_build().file(&generated_cpp).compile("zngur_generated");

    Ok(())
}
