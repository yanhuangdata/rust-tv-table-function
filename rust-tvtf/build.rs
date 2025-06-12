use anyhow::Context;
use std::{path::PathBuf, process::Command};
use zngur::Zngur;

fn main() -> anyhow::Result<()> {
    if std::env::var("TARGET").unwrap().contains("apple-darwin") {
        println!("cargo::rustc-link-arg-cdylib=-Wl,-install_name,@rpath/librust_tvtf.dylib");
    }

    println!("cargo:rerun-if-changed=main.zng");
    println!("cargo:rerun-if-changed=../ci/scripts/fix_generated.sh");
    println!("cargo:rerun-if-changed=../zngur/extra_generated.h");
    let cwd = std::env::current_dir().context("failed to get current dir")?;
    let root_dir = {
        let size = cwd.components().count();
        cwd.components()
            .take(size.wrapping_sub(1))
            .collect::<PathBuf>()
    };
    let cpp = root_dir.join("cpp");
    let cpp_zngur = cpp.join("zngur");
    std::fs::create_dir_all(&cpp_zngur).context("Failed to create cpp zngur dir")?;
    let generated_header = cpp_zngur.join("generated.h");
    let extra_generated_header = cpp_zngur.join("extra_generated.h");
    let generated_cpp = root_dir.join("cpp/zngur/generated.cpp");
    Zngur::from_zng_file(cwd.join("main.zng"))
        .with_cpp_file(&generated_cpp)
        .with_h_file(&generated_header)
        .with_rs_file(cwd.join("src/zngur_generated.rs"))
        .generate();

    let zngur_tvtf = cpp.join("src").join("zngur_tvtf");
    std::fs::create_dir_all(&zngur_tvtf).context("Failed to create zngur_tvtf dir")?;
    std::fs::copy(&generated_header, zngur_tvtf.join("generated.h"))
        .context("Failed to copy generated.h")?;
    std::fs::copy(
        &extra_generated_header,
        zngur_tvtf.join("extra_generated.h"),
    )
    .context("Failed to copy extra_generated.h")?;

    let output = Command::new(root_dir.join("ci/scripts/fix_generated.sh"))
        .current_dir(&root_dir)
        .output()
        .context("failed to execute script")?;

    if std::env::var("CARGO_DEBUG").is_ok() {
        println!("cargo:warning=fix_generated.sh output:");
        println!("cargo:warning={}", String::from_utf8_lossy(&output.stdout));
        println!("cargo:warning={}", String::from_utf8_lossy(&output.stderr));
        println!("cargo:warning={}", &output.status);
    }

    Ok(())
}
