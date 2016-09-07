#include <agency/execution/executor/detail/new_executor_traits.hpp>
#include <type_traits>
#include <iostream>

#include "test_executors.hpp"

struct bulk_executor_without_shape_type
{
  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory(size_t)>::type
  bulk_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory);
};

struct bulk_executor_with_shape_type
{
  struct shape_type
  {
    size_t n;
  };

  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory(shape_type)>::type
  bulk_execute(Function f, shape_type n, ResultFactory result_factory, SharedFactory shared_factory);
};

int main()
{
  using namespace agency::detail::new_executor_traits_detail;

  static_assert(!agency::detail::is_detected<executor_shape_t, not_an_executor>::value, "executor_shape_t<not_an_executor> should not be detected");

  static_assert(agency::detail::is_detected_exact<size_t, executor_shape_t, bulk_executor_without_shape_type>::value, "bulk_executor_without_shape_type should have size_t shape_type");

  static_assert(agency::detail::is_detected_exact<bulk_executor_with_shape_type::shape_type, executor_shape_t, bulk_executor_with_shape_type>::value, "bulk_executor_with_shape_type should have bulk_executor_with_shape_type::shape_type shape_type");

  static_assert(agency::detail::is_detected_exact<size_t, executor_shape_t, bulk_continuation_executor>::value, "bulk_continuation_executor should have size_t shape_type");

  std::cout << "OK" << std::endl;

  return 0;
}

