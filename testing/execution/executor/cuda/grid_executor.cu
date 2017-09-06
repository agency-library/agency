#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/cuda/memory/allocator.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/member_barrier_type_or.hpp>

int main()
{
  using namespace agency;

  // XXX fill out this function with appropriate unit tests

  static_assert(is_executor<cuda::grid_executor>::value,
    "grid_executor should be an executor");

  static_assert(detail::is_detected_exact<cuda::async_future<int>, executor_future_t, cuda::grid_executor, int>::value,
    "grid_executor should have cuda::async_future future type");

  static_assert(detail::is_detected_exact<cuda::allocator<int>, executor_allocator_t, cuda::grid_executor, int>::value,
    "grid_executor should have cuda::allocator type");

  static_assert(detail::is_detected_exact<detail::scoped_in_place_type_t<void, cuda::detail::block_barrier>, detail::member_barrier_type_or_t, cuda::grid_executor, void>::value,
    "grid_executor::barrier_type should be scoped_in_place_type_t<void, cuda::detail::block_barrier");

  std::cout << "OK" << std::endl;

  return 0;
}

