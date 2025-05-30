#include "../zngur/generated.h"
#include "rust_tv_table_function.h"

#include <boost/algorithm/string/predicate.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

TEST_CASE("call rust") {
  const auto registries = rust::crate::get_function_registries();
  for (std::size_t i = 0; i < registries.len(); i++) {
    const auto registry = registries.get(i).unwrap();
    const auto name = registry.name();
    const auto *ptr = name.as_ptr();
    const auto name_view = std::string_view{reinterpret_cast<const char *>(ptr), name.len()};
    if (!boost::iequals(name_view, "addtotals")) {
      continue;
    }
    auto table_func = rust::crate::create_raw(registry, nullptr,
                                              reinterpret_cast<int8_t const *>("Asia/Shanghai"))
                          .unwrap();
    REQUIRE(name_view == "addtotals");
  }
}
