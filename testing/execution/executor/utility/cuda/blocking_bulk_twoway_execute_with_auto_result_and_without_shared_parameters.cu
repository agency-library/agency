#include <agency/agency.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <agency/cuda.hpp>
#include <iostream>

#include "../../test_executors.hpp"


template<class Executor>
void test_returning_void(Executor exec)
{
  agency::executor_shape_t<Executor> shape{100};
  
  size_t increment_me = 0;
  std::mutex mut;
  agency::detail::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(exec, [&](size_t)
  {
    mut.lock();
    increment_me += 1;
    mut.unlock();
  },
  shape);
  
  assert(increment_me == shape);
}


template<class Executor>
void test_returning_results(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  size_t shape = 10;
  
  auto result = agency::detail::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(exec, [](index_type)
  {
    return 13;
  },
  shape);
  
  using container_type = agency::vector<int, agency::executor_allocator_t<Executor,int>>;
  assert(container_type(shape, 13) == result);
}


__managed__ int increment_me;


template<class Executor>
void test_returning_void2(Executor exec)
{
  agency::executor_shape_t<Executor> shape{10,10};
  using index_type = agency::executor_index_t<Executor>;
  
  increment_me = 0;
  agency::detail::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(exec, [] __device__ (index_type)
  {
    atomicAdd(&increment_me, 1);
  },
  shape);
  
  assert(increment_me == 10 * 10);
}


template<class Executor>
void test_returning_results2(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape{10,10};
  
  auto result = agency::detail::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(exec, [] __host__ __device__ (index_type)
  {
    return 13;
  },
  shape);
  
  using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<Executor,int>>;
  assert(container_type(shape, 13) == result);
}


int main()
{
  test_returning_void(bulk_twoway_executor());
  test_returning_void(bulk_then_executor());
  test_returning_void(not_a_bulk_twoway_executor());
  test_returning_void(not_a_bulk_then_executor());
  test_returning_void(complete_bulk_executor());

  test_returning_results(bulk_twoway_executor());
  test_returning_results(bulk_then_executor());
  test_returning_results(not_a_bulk_twoway_executor());
  test_returning_results(not_a_bulk_then_executor());
  test_returning_results(complete_bulk_executor());

  test_returning_void2(agency::cuda::grid_executor());
  test_returning_results2(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;
  
  return 0;
}

