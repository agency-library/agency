#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/future.hpp>
#include <agency/execution/executor/detail/new_executor_traits.hpp>
#include <agency/cuda.hpp>

#include "../test_executors.hpp"


template<class Executor>
void test_with_non_void_predecessor(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  auto predecessor_future = agency::executor_traits<Executor>::template make_ready_future<int>(exec, 7);
  
  auto f = then_execute(exec, [] __host__ __device__ (int& predecessor)
  {
    return predecessor + 13;
  },
  predecessor_future);
  
  auto result = f.get();
  
  assert(7 + 13 == result);
}


template<class Executor>
void test_with_void_predecessor(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  auto predecessor_future = agency::executor_traits<Executor>::template make_ready_future<void>(exec);

  auto f = then_execute(exec, [] __host__ __device__ { return 13; }, predecessor_future);
  
  auto result = f.get();
  
  assert(13 == result);
}


int main()
{
  test_with_non_void_predecessor(continuation_executor());
  test_with_non_void_predecessor(bulk_continuation_executor());

  test_with_non_void_predecessor(agency::cuda::grid_executor());

  test_with_void_predecessor(continuation_executor());
  test_with_void_predecessor(bulk_continuation_executor());

  test_with_void_predecessor(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;

  return 0;
}
