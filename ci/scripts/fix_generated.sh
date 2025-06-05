#!/usr/bin/env bash

set -x # enable debugging

# Function to determine the OS
get_os() {
    case "$(uname -s)" in
        Darwin*)    echo "macOS" ;;
        Linux*)     echo "Linux" ;;
        *)          echo "Unknown" ;;
    esac
}
# Read the first argument as gen_file, if not provided, use `generated.h` as the default value
gen_file=${1:-cpp/zngur/generated.h}
echo "[fixing generated zngur files] cpp_header=${gen_file}"
OS=$(get_os)
echo "fixing ${gen_file}"
line=$(grep -n 'inline uint8_t\* __zngur_internal_data_ptr<int8_t>(const int8_t& t)' ${gen_file} | cut -d':' -f1)
insert_line=$((line - 2))
if [ "$OS" = "macOS" ]; then
    sed -i '' "${insert_line}a\\
 #include \"extra_generated.h\"
" ${gen_file}
elif [ "$OS" = "Linux" ]; then
    sed -i "${insert_line}a #include \"extra_generated.h\"" ${gen_file}
else
    echo "Unsupported OS"
    exit 1
fi
