# Contributing

## Development Setup

- Install the Rust toolchain specified in `rust-toolchain.toml` (Rust 1.90 with `rustfmt` and `clippy`).
- Use the workspace commands from the repository root:
  - `cargo fmt` – formats sources.
  - `cargo clippy --tests` – lints the code.
    - `cargo clippy --fix --tests` – for automatically fixes.
  - `cargo test --workspace` – executes the full test suite.

## Adding a New Table Function

1. **Create the implementation** under `rust-tvtf/src/funcs/`.
    - Follow an existing function (for example `addtotals.rs`) as a template.
    - Provide a constructor that parses `Args`/named arguments as needed.
    - Implement the `TableFunction` trait from `rust_tvtf_api` and include focused unit tests in the same file.
2. **Expose the module** in `rust-tvtf/src/funcs/mod.rs` by adding `pub mod <name>;` and `pub use <name>::*;`
   statements.
3. **Register the function** in `rust-tvtf/src/lib.rs` by adding a `FunctionRegistry::builder()` entry:
    - Use a snake_case function name.
    - Wire the constructor inside the `init` closure and define the accepted signature parameters.
    - Set `require_ordered(true)` only if the function needs ordered input.
4. **Update public APIs** if necessary (for example, expose new argument types in `rust-tvtf-api`).
5. **Format, lint, and test** the workspace:

   ```sh
   cargo fmt
   cargo clippy --workspace --all-targets --all-features
   cargo test --workspace
   ```

6. Submit a pull request with a description of the new table function and its usage.

## Release Process

1. **Prepare the release commit**:
    - Bump the version numbers in each crate (`rust-tvtf`, `rust-tvtf-api`, `rust-tvtf-example`) and update `Cargo.lock`
      with `cargo update -p <crate>` or `cargo check`.
    - Ensure CI passes on the release branch.

2. **Merge to `main`** (or the release branch) and fetch the latest changes locally.
3. **Tag the release** with the new semantic version:

   ```sh
   git tag -a <version> -m "release <version>"
   git push origin <version>
   ```

4. **Wait for GitHub Actions**:
   - The `Publish releases to GitHub` workflow builds release artifacts and creates a **draft** release tied to the
      tag.
   - Monitor the workflow run until it finishes successfully.

5. **Publish the GitHub release**:
   - Open the draft release, review the notes/assets, and click **Publish release**.
   - Publishing triggers the `Sync Release to GitLab` workflow, which uploads the assets to GitLab automatically.

6. **Verify the mirrors**: confirm that the GitHub release is live and that the GitLab sync workflow completed without
   errors.
