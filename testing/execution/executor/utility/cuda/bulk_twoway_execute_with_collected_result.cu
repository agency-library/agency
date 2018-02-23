#include <agency/agency.hpp>
#include <agency/experimental/ndarray.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <agency/cuda.hpp>
#include <iostream>

#include "../../test_executors.hpp"

template<class Executor>
void test(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = agency::detail::bulk_twoway_execute_with_collected_result(exec,
    [](index_type idx, std::vector<int>& shared_arg)
    {
      return shared_arg[idx];
    },
    shape,
    [=]{ return std::vector<int>(shape); },    // results
    [=]{ return std::vector<int>(shape, 13); } // shared_arg
  );

  auto result = f.get();
  
  assert(std::vector<int>(shape, 13) == result);
}


template<class Executor>
void test2(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape{10,10};

  using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<Executor,int>>;
  
  auto f = agency::detail::bulk_twoway_execute_with_collected_result(exec,
    [] __host__ __device__ (index_type, int& outer_shared_arg, int& inner_shared_arg)
    {
      return outer_shared_arg + inner_shared_arg;
    },
    shape,
    [=] __host__ __device__ { return container_type(shape); }, // results
    [] __host__ __device__  { return 7; },                     // outer_shared_arg
    [] __host__ __device__  { return 13; }                     // inner_shared_arg
  );

  auto result = f.get();
  
  assert(container_type(shape, 7 + 13) == result);
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

