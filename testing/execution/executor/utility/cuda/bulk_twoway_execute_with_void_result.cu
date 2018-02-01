#include <agency/agency.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <agency/cuda.hpp>
#include <cassert>
#include <iostream>

#include "../../test_executors.hpp"

template<class Executor>
void test(Executor exec)
{
  std::atomic<int> counter{0};

  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape{10};
  
  auto f = agency::detail::bulk_twoway_execute_with_void_result(exec,
    [&](index_type, int& shared_arg)
    {
      counter += shared_arg;
    },
    shape,
    []{ return 13; } // shared_arg
  );

  f.wait();
  
  assert(counter == 13 * 10);
}


__managed__ int increment_me;

template<class Executor>
void test2(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape{10,10};

  increment_me = 0;
  
  auto f = agency::detail::bulk_twoway_execute_with_void_result(exec,
    [] __device__ (index_type, int& outer_arg, int& inner_arg)
    {
      atomicAdd(&increment_me, outer_arg + inner_arg);
    },
    shape,
    [] __host__ __device__ { return 7; }, // outer_arg
    [] __host__ __device__ { return 13; } // inner_arg
  );

  f.wait();
  
  assert(increment_me == (7 + 13) * 10 * 10);
}


int main()
{
  test(bulk_twoway_executor());
  test(bulk_then_executor());
  test(not_a_bulk_twoway_executor());
  test(not_a_bulk_then_executor());
  test(complete_bulk_executor());

  test2(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

