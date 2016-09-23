#include <agency/agency.hpp>
#include <agency/execution/executor/detail/customization_points.hpp>
#include <agency/cuda.hpp>
#include <iostream>

#include "../../test_executors.hpp"


template<class Executor>
void test_returning_void(Executor exec)
{
  using namespace agency::detail::executor_customization_points_detail;

  agency::new_executor_shape_t<Executor> shape{100};
  
  int increment_me = 0;
  std::mutex mut;
  bulk_execute_with_auto_result_and_without_shared_parameters(exec, [&](size_t idx)
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
  using namespace agency::detail::executor_customization_points_detail;

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  size_t shape = 10;
  
  auto result = bulk_execute_with_auto_result_and_without_shared_parameters(exec, [](index_type idx)
  {
    return 13;
  },
  shape);
  
  using container_type = agency::new_executor_container_t<Executor,int>;
  assert(container_type(shape, 13) == result);
}


__managed__ int increment_me;


template<class Executor>
void test_returning_void2(Executor exec)
{
  using namespace agency::detail::executor_customization_points_detail;

  agency::new_executor_shape_t<Executor> shape{10,10};
  using index_type = agency::new_executor_index_t<Executor>;
  
  increment_me = 0;
  bulk_execute_with_auto_result_and_without_shared_parameters(exec, [] __device__ (index_type idx)
  {
    atomicAdd(&increment_me, 1);
  },
  shape);
  
  assert(increment_me == 10 * 10);
}


template<class Executor>
void test_returning_results2(Executor exec)
{
  using namespace agency::detail::executor_customization_points_detail;

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  shape_type shape{10,10};
  
  auto result = bulk_execute_with_auto_result_and_without_shared_parameters(exec, [] __host__ __device__ (index_type idx)
  {
    return 13;
  },
  shape);
  
  using container_type = agency::new_executor_container_t<Executor,int>;
  assert(container_type(shape, 13) == result);
}


int main()
{
  test_returning_void(bulk_synchronous_executor());
  test_returning_void(bulk_asynchronous_executor());
  test_returning_void(bulk_continuation_executor());
  test_returning_void(not_a_bulk_synchronous_executor());
  test_returning_void(not_a_bulk_asynchronous_executor());
  test_returning_void(not_a_bulk_continuation_executor());
  test_returning_void(complete_bulk_executor());

  test_returning_results(bulk_synchronous_executor());
  test_returning_results(bulk_asynchronous_executor());
  test_returning_results(bulk_continuation_executor());
  test_returning_results(not_a_bulk_synchronous_executor());
  test_returning_results(not_a_bulk_asynchronous_executor());
  test_returning_results(not_a_bulk_continuation_executor());
  test_returning_results(complete_bulk_executor());

  test_returning_void2(agency::cuda::grid_executor());
  test_returning_results2(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;
  
  return 0;
}

