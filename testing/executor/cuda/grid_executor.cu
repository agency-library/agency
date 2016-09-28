#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/cuda/memory/allocator.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>

int main()
{
  using namespace agency;

  // XXX fill out this function with appropriate unit tests

  static_assert(detail::is_detected_exact<cuda::async_future<int>, executor_future_t, cuda::grid_executor, int>::value,
    "grid_executor should have cuda::async_future future type");

  static_assert(detail::is_detected_exact<cuda::allocator<int>, executor_allocator_t, cuda::grid_executor, int>::value,
    "grid_executor should have cuda::allocator type");

  std::cout << "OK" << std::endl;

  return 0;
}

