#!/usr/bin/env bash

set -e # exit on error
set -x # enable debugging

# Function to determine the OS
get_os() {
    case "$(uname -s)" in
        Darwin*)    echo "macOS" ;;
        Linux*)     echo "Linux" ;;
        *)          echo "Unknown" ;;
    esac
}

# Function to get library extension based on OS
get_lib_extension() {
    case "$(get_os)" in
        macOS)      echo "dylib" ;;
        Linux)      echo "so" ;;
        *)          echo "so" ;;
    esac
}

# Parse command line arguments
BUILD_TARGET=${1:-""}
LIB_NAME=${2:-"rust_tvtf"}
VERSION=${3:-"dev"}
INCLUDE_FILES=${4:-""}

if [ -z "$BUILD_TARGET" ]; then
    echo "Usage: $0 <build-target> [lib-name] [version] [include-files]"
    echo "Example: $0 x86_64-unknown-linux-gnu rust_tvtf v1.0.0 'cpp/zngur/generated/generated.h,cpp/zngur/generated/extra_generated.h'"
    exit 1
fi

echo "Building and packaging release..."
echo "Build target: $BUILD_TARGET"
echo "Library name: $LIB_NAME"
echo "Version: $VERSION"
echo "Include files: $INCLUDE_FILES"

# Build the project for the specified target
echo "Building project..."
cargo build --release --target "$BUILD_TARGET"

# Determine library file extension and path
LIB_EXT=$(get_lib_extension)
LIB_FILE="lib${LIB_NAME}.${LIB_EXT}"
LIB_PATH="./target/${BUILD_TARGET}/release/${LIB_FILE}"

# Check if library exists
if [ ! -f "$LIB_PATH" ]; then
    echo "Error: Library file not found at $LIB_PATH"
    echo "Available files in target/${BUILD_TARGET}/release/:"
    ls -la "./target/${BUILD_TARGET}/release/" || echo "Release directory not found"
    exit 1
fi

# Create temporary directory for packaging
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="${TEMP_DIR}/${LIB_NAME}-${VERSION}-${BUILD_TARGET}"
mkdir -p "$PACKAGE_DIR"

echo "Packaging files..."

# Copy the library file
cp "$LIB_PATH" "$PACKAGE_DIR/"
echo "Copied library: $LIB_FILE"

# Copy additional include files if specified
if [ -n "$INCLUDE_FILES" ]; then
    IFS=',' read -ra FILES <<< "$INCLUDE_FILES"
    for file in "${FILES[@]}"; do
        # Remove leading/trailing whitespace
        file=$(echo "$file" | xargs)
        if [ -f "$file" ]; then
            # Create directory structure in package if needed
            target_dir="$PACKAGE_DIR/$(dirname "$file")"
            mkdir -p "$target_dir"
            cp "$file" "$target_dir/"
            echo "Copied include file: $file"
        else
            echo "Warning: Include file not found: $file"
        fi
    done
fi

# Create archive name
ARCHIVE_NAME="${LIB_NAME}-${VERSION}-${BUILD_TARGET}.tar.gz"

# Create tar.gz archive
echo "Creating archive: $ARCHIVE_NAME"
cd "$TEMP_DIR"
tar -czf "$ARCHIVE_NAME" "${LIB_NAME}-${VERSION}-${BUILD_TARGET}"

# Move archive to current directory
mv "$ARCHIVE_NAME" "$OLDPWD/"
cd "$OLDPWD"

# Generate SHA256 checksum
echo "Generating checksum..."
CHECKSUM_FILE="${ARCHIVE_NAME}.sha256"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$ARCHIVE_NAME" > "$CHECKSUM_FILE"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$ARCHIVE_NAME" > "$CHECKSUM_FILE"
else
    echo "Warning: No SHA256 utility found, skipping checksum generation"
fi

# Clean up
rm -rf "$TEMP_DIR"

echo "Package created successfully:"
echo "  Archive: $ARCHIVE_NAME"
if [ -f "$CHECKSUM_FILE" ]; then
    echo "  Checksum: $CHECKSUM_FILE"
    echo "  SHA256: $(cat "$CHECKSUM_FILE")"
fi

echo "Contents of the archive:"
tar -tzf "$ARCHIVE_NAME"

echo "Release packaging completed!"
