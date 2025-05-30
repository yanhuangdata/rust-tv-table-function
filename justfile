#!/usr/bin/env just --justfile

dylib_ext := if os() == "macos" { "dylib" } else { "so" }

set dotenv-load := true
default_build_type := env_var_or_default("BUILD_TYPE", "debug")

# thrift.rs/format.rs are copied from arrow-rs parquet and we don't want to include them in coverage report
llvm_cov_ignore_files := env_var_or_default("LLVM_COV_IGNORE_FILES", ".*/bloom_filter/thrift.rs|.*/bloom_filter/format.rs|.*/zngur_generated.rs")

# switch to x64_toolchain
x64_toolchain:
    rustup override set 1.79.0-x86_64-apple-darwin
    rustup override set 1.79.0-x86_64-apple-darwin --path vector
    rustup override list

# by default, it will build a shared library in debug mode
# use `just build release` to build a static release library
build build_type="":
  cargo build {{ if build_type == "release" { "--release" } else { "" } }}
  ln -sf {{ justfile_directory() }}/target/{{ if build_type == "release" { "release" } else { "debug" } }}/librust_tvtf.{{dylib_ext}} cpp/zngur/librust_tvtf.{{dylib_ext}}

check_format:
    cargo fmt --check

# generate coverage report
# by default, it will generate a cobertura report
# you can specify the format by passing the format argument like this: `just test --html`
test format="--html":
  cargo llvm-cov --ignore-filename-regex="{{llvm_cov_ignore_files}}" {{format}} nextest --workspace --config-file=./config/nextest.toml || exit 1 && cd ..

test_html:
  just test --html && open target/llvm-cov/html/index.html

test_cobertura:
  just test "--cobertura --output-path=target/cobertura.xml"

clean:
  cargo clean
