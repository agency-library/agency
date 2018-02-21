#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/future.hpp>
#include <agency/container/vector.hpp>
#include <agency/experimental/ndarray.hpp>
#include <agency/execution/executor/detail/execution_functions/bulk_then_execute.hpp>
#include <agency/cuda.hpp>

#include "../../test_executors.hpp"


template<class Executor>
void test_with_non_void_predecessor(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<int>(exec, 7);

  using index_type = agency::executor_index_t<Executor>;
  using int_vector = agency::vector<int, agency::executor_allocator_t<Executor,int>>;

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute(exec,
    [] __host__ __device__ (index_type idx, int& predecessor, int_vector& results, int_vector& shared_arg)
    {
      results[idx] = predecessor + shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=] __host__ __device__ { return int_vector(shape); },     // results
    [=] __host__ __device__ { return int_vector(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  assert(int_vector(10, 7 + 13) == result);
}


template<class Executor>
void test_with_void_predecessor(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<void>(exec);

  using index_type = agency::executor_index_t<Executor>;
  using int_vector = agency::vector<int, agency::executor_allocator_t<Executor,int>>;

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute(exec,
    [] __host__ __device__ (index_type idx, int_vector& results, int_vector& shared_arg)
    {
      results[idx] = shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=] __host__ __device__ { return int_vector(shape); },     // results
    [=] __host__ __device__ { return int_vector(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  assert(int_vector(10, 13) == result);
}


template<class TwoLevelExecutor>
void test_with_non_void_predecessor2(TwoLevelExecutor exec)
{
  auto predecessor_future = agency::make_ready_future<int>(exec, 7);

  using shape_type = agency::executor_shape_t<TwoLevelExecutor>;
  using index_type = agency::executor_index_t<TwoLevelExecutor>;

  shape_type shape{10,10};

  using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<TwoLevelExecutor,int>>;
  
  auto f = agency::detail::bulk_then_execute(exec,
    [] __host__ __device__ (index_type idx, int& predecessor, container_type& results, int& outer_shared_arg, int& inner_shared_arg)
    {
      results[idx] = predecessor + outer_shared_arg + inner_shared_arg;
    },
    shape,
    predecessor_future,
    [=] __host__ __device__ { return container_type(shape); }, // results
    [] __host__ __device__ { return 13; },                     // outer_shared_arg
    [] __host__ __device__ { return 42; }                      // inner_shared_arg
  );
  
  auto result = f.get();
  
  assert(container_type(shape, 7 + 13 + 42) == result);
}


template<class TwoLevelExecutor>
void test_with_void_predecessor2(TwoLevelExecutor exec)
{
  auto predecessor_future = agency::make_ready_future<void>(exec);

  using shape_type = agency::executor_shape_t<TwoLevelExecutor>;
  using index_type = agency::executor_index_t<TwoLevelExecutor>;

  shape_type shape{10,10};

  using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<TwoLevelExecutor,int>>;
  
  auto f = agency::detail::bulk_then_execute(exec,
    [] __host__ __device__ (index_type idx, container_type& results, int& outer_shared_arg, int& inner_shared_arg)
    {
      results[idx] = outer_shared_arg + inner_shared_arg;
    },
    shape,
    predecessor_future,
    [=] __host__ __device__ { return container_type(shape); }, // results
    [] __host__ __device__ { return 13; },                     // outer_shared_arg
    [] __host__ __device__ { return 42; }                      // inner_shared_arg
  );
  
  auto result = f.get();
  
  assert(container_type(shape, 13 + 42) == result);
}


int main()
{
  test_with_non_void_predecessor(bulk_twoway_executor());
  test_with_non_void_predecessor(bulk_then_executor());
  test_with_non_void_predecessor(not_a_bulk_twoway_executor());
  test_with_non_void_predecessor(not_a_bulk_then_executor());
  test_with_non_void_predecessor(complete_bulk_executor());

  test_with_void_predecessor(bulk_twoway_executor());
  test_with_void_predecessor(bulk_then_executor());
  test_with_void_predecessor(not_a_bulk_twoway_executor());
  test_with_void_predecessor(not_a_bulk_then_executor());
  test_with_void_predecessor(complete_bulk_executor());

  test_with_non_void_predecessor(agency::flattened_executor<agency::cuda::grid_executor>());
  test_with_void_predecessor(agency::flattened_executor<agency::cuda::grid_executor>());

  test_with_non_void_predecessor2(agency::cuda::grid_executor());
  test_with_void_predecessor2(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

