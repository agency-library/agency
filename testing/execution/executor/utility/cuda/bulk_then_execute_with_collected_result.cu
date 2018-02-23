#include <agency/agency.hpp>
#include <agency/experimental/ndarray.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <agency/cuda.hpp>
#include <iostream>

#include "../../test_executors.hpp"

template<class Executor>
void test_with_void_predecessor(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  auto predecessor_future = agency::make_ready_future<void>(exec);

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute_with_collected_result(exec,
    [](index_type idx, std::vector<int>& shared_arg)
    {
      return shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape); },    // results
    [=]{ return std::vector<int>(shape, 13); } // shared_arg
  );

  auto result = f.get();
  
  assert(std::vector<int>(shape, 13) == result);
}


template<class Executor>
void test_with_non_void_predecessor(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  auto predecessor_future = agency::make_ready_future<int>(exec, 7);

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute_with_collected_result(exec,
    [](index_type idx, int& predecessor, std::vector<int>& shared_arg)
    {
      return predecessor + shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape); },    // results
    [=]{ return std::vector<int>(shape, 13); } // shared_arg
  );

  auto result = f.get();
  
  assert(std::vector<int>(shape, 7 + 13) == result);
}


template<class Executor>
void test_with_void_predecessor2(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<void>(exec);

  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape{10,10};

  using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<Executor,int>>;
  
  auto f = agency::detail::bulk_then_execute_with_collected_result(exec,
    [] __host__ __device__ (index_type, int& outer_shared_arg, int& inner_shared_arg)
    {
      return outer_shared_arg + inner_shared_arg;
    },
    shape,
    predecessor_future,
    [=] __host__ __device__ { return container_type(shape); }, // results
    [] __host__ __device__ { return 7; },                      // outer_shared_arg
    [] __host__ __device__ { return 13; }                      // inner_shared_arg
  );
  
  auto result = f.get();
  
  assert(container_type(shape, 7 + 13) == result);
}


template<class Executor>
void test_with_non_void_predecessor2(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<int>(exec, 42);

  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape{10,10};

  using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<Executor,int>>;
  
  auto f = agency::detail::bulk_then_execute_with_collected_result(exec,
    [] __host__ __device__ (index_type, int& predecessor, int& outer_shared_arg, int& inner_shared_arg)
    {
      return predecessor + outer_shared_arg + inner_shared_arg;
    },
    shape,
    predecessor_future,
    [=] __host__ __device__ { return container_type(shape); }, // results
    [] __host__ __device__ { return 7; },                      // outer_shared_arg
    [] __host__ __device__ { return 13; }                      // inner_shared_arg
  );
  
  auto result = f.get();
  
  assert(container_type(shape, 42 + 7 + 13) == result);
}


int main()
{
  test_with_void_predecessor(bulk_twoway_executor());
  test_with_void_predecessor(bulk_then_executor());
  test_with_void_predecessor(not_a_bulk_twoway_executor());
  test_with_void_predecessor(not_a_bulk_then_executor());
  test_with_void_predecessor(complete_bulk_executor());

  test_with_non_void_predecessor(bulk_twoway_executor());
  test_with_non_void_predecessor(bulk_then_executor());
  test_with_non_void_predecessor(not_a_bulk_twoway_executor());
  test_with_non_void_predecessor(not_a_bulk_then_executor());
  test_with_non_void_predecessor(complete_bulk_executor());

  test_with_void_predecessor2(agency::cuda::grid_executor());

  test_with_non_void_predecessor2(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

