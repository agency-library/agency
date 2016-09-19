#include <agency/execution/executor/detail/new_executor_traits.hpp>
#include <agency/cuda.hpp>
#include <type_traits>
#include <iostream>

struct not_an_executor {};

struct bulk_executor_without_index_type
{
  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory()>::type
  bulk_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory);
};

struct bulk_executor_with_shape_type_without_index_type
{
  struct shape_type
  {
    size_t n;
  };

  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory()>::type
  bulk_execute(Function f, shape_type n, ResultFactory result_factory, SharedFactory shared_factory);
};

struct bulk_executor_with_index_type
{
  struct index_type
  {
    size_t i;
  };

  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory()>::type
  bulk_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory);
};

struct bulk_executor_with_shape_type_with_index_type
{
  struct shape_type
  {
    size_t n;
  };

  struct index_type
  {
    size_t i;
  };

  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory()>::type
  bulk_execute(Function f, shape_type n, ResultFactory result_factory, SharedFactory shared_factory);
};

int main()
{
  using namespace agency::detail::new_executor_traits_detail;

  static_assert(!agency::detail::is_detected<executor_index_t, not_an_executor>::value,
    "executor_index_t<not_an_executor> should not be detected");

  static_assert(agency::detail::is_detected_exact<size_t, executor_index_t, bulk_executor_without_index_type>::value,
    "bulk_executor_without_index_type should have size_t index_type");

  static_assert(agency::detail::is_detected_exact<bulk_executor_with_shape_type_without_index_type::shape_type, executor_index_t, bulk_executor_with_shape_type_without_index_type>::value,
    "bulk_executor_with_shape_type_without_index_type should have bulk_executor_with_shape_type_without_index_type::shape_type index_type");

  static_assert(agency::detail::is_detected_exact<bulk_executor_with_index_type::index_type, executor_index_t, bulk_executor_with_index_type>::value,
    "bulk_executor_with_index_type should have bulk_executor_with_index_type::index_type index_type");

  static_assert(agency::detail::is_detected_exact<bulk_executor_with_shape_type_with_index_type::index_type, executor_index_t, bulk_executor_with_shape_type_with_index_type>::value,
    "bulk_executor_with_shape_type_with_index_type should have bulk_executor_with_shape_type_with_index_type::index_type index_type");

  static_assert(agency::detail::is_detected_exact<agency::uint2, executor_index_t, agency::cuda::grid_executor>::value,
    "grid_executor should have uint2 index_type");

  std::cout << "OK" << std::endl;

  return 0;
}
