#include <agency/execution/executor/detail/new_executor_traits.hpp>
#include <agency/cuda.hpp>
#include <type_traits>
#include <iostream>

#include "../test_executors.hpp"

int main()
{
  using namespace agency::detail::new_executor_traits_detail;

  static_assert(!agency::detail::is_detected<executor_shape_t, not_an_executor>::value, "executor_shape_t<not_an_executor> should not be detected");

  static_assert(agency::detail::is_detected_exact<size_t, executor_shape_t, bulk_executor_without_shape_type>::value, "bulk_executor_without_shape_type should have size_t shape_type");

  static_assert(agency::detail::is_detected_exact<bulk_executor_with_shape_type::shape_type, executor_shape_t, bulk_executor_with_shape_type>::value, "bulk_executor_with_shape_type should have bulk_executor_with_shape_type::shape_type shape_type");

  static_assert(agency::detail::is_detected_exact<size_t, executor_shape_t, bulk_continuation_executor>::value, "bulk_continuation_executor should have size_t shape_type");

  static_assert(agency::detail::is_detected_exact<agency::uint2, executor_shape_t, agency::cuda::grid_executor>::value, "grid_executor should have uint2 shape_type");

  std::cout << "OK" << std::endl;

  return 0;
}

