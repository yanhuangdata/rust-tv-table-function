#include "../zngur/generated.h"
#include "rust_tv_table_function.h"
#include <dlfcn.h>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

std::string _get_lib_ext() {
#if defined(__APPLE__)
  return ".dylib";
#elif defined(__linux__)
  return ".so";
#else
#error "Unsupported operating system"
#endif
}

TEST_CASE("call rust") {
  auto dylib_path = "../zngur/librust_tvtf" + _get_lib_ext();
  REQUIRE(std::filesystem::exists(dylib_path));
  void *dylib = dlopen(dylib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  auto api = __zngur_dyn_api{dylib};
  std::string error_msg;
  REQUIRE(api.init(error_msg));
  auto registries_result = rust::crate::get_function_registries(&api);
  if (registries_result.is_err(&api)) {
    FAIL("Failed to get function registries");
  }
  auto registries = registries_result.unwrap(&api);
  for (std::size_t i = 0; i < registries.len(&api); i++) {
    const auto registry = registries.get(&api, i).unwrap(&api);
    const auto name = registry.name(&api);
    auto sig_result = registry.signatures(&api);
    if (sig_result.is_err(&api)) {
      FAIL("Failed to get signatures for registry");
    }
    const auto *ptr = name.as_ptr(&api);
    const auto name_view = std::string_view{reinterpret_cast<const char *>(ptr), name.len(&api)};
    if (name_view != "addtotals") {
      continue;
    }
    auto sig = sig_result.unwrap(&api);
    auto sig_view = std::string_view{reinterpret_cast<const char *>(sig.as_str(&api).as_ptr(&api)),
                                     sig.as_str(&api).len(&api)};
    auto require_ordered = registry.require_ordered(&api);
    REQUIRE(require_ordered == false);
    REQUIRE(
        sig_view ==
        R"([{"parameters":[]},{"parameters":[{"name":null,"default":null,"arg_type":"INT"}]}])");
    auto table_func =
        rust::rust_tvtf_api::create_raw(&api, registry, nullptr, nullptr,
                                        reinterpret_cast<int8_t const *>("Asia/Shanghai"))
            .unwrap(&api);
    REQUIRE(name_view == "addtotals");
  }
}
