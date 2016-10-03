#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/future.hpp>
#include <agency/execution/executor/customization_points.hpp>
#include <agency/cuda.hpp>

#include "../../test_executors.hpp"


template<class Executor>
void test(Executor exec)
{
  auto f = agency::async_execute(exec, [] __host__ __device__ { return 7;});
  
  auto result = f.get();
  
  assert(7 == result);
}


int main()
{
  test(continuation_executor());
  test(asynchronous_executor());
  test(bulk_continuation_executor());
  test(bulk_asynchronous_executor());
  test(agency::cuda::grid_executor());

  // XXX call test() with all the other types of executors

  std::cout << "OK" << std::endl;

  return 0;
}

