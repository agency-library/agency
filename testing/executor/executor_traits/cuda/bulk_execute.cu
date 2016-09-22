#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/future.hpp>
#include <agency/execution/executor/detail/new_executor_traits.hpp>
#include <agency/cuda.hpp>

#include "../test_executors.hpp"


template<class Executor>
void test(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  shape_type shape = 10;
  
  auto result = bulk_execute(exec,
    [](index_type idx, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = 7 + shared_arg[idx];
    },
    shape,
    [=]{ return std::vector<int>(shape); },     // results
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  assert(std::vector<int>(10, 7 + 13) == result);
}


template<class TwoLevelExecutor>
void test2(TwoLevelExecutor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  using shape_type = agency::new_executor_shape_t<TwoLevelExecutor>;
  using index_type = agency::new_executor_index_t<TwoLevelExecutor>;

  shape_type shape{10,10};

  using container_type = executor_container_t<TwoLevelExecutor, int>;
  
  auto result = bulk_execute(exec,
    [] __device__ (index_type idx, container_type& results, int& outer_shared_arg, int& inner_shared_arg)
    {
      results[idx] = outer_shared_arg + inner_shared_arg;
    },
    shape,
    [=] __host__ __device__ { return container_type(shape); }, // results
    [] __host__ __device__ { return 7; },                      // outer_shared_arg
    [] __host__ __device__ { return 13; }                      // inner_shared_arg
  );
  
  assert(container_type(10, 7 + 13) == result);
}


int main()
{
  test(bulk_synchronous_executor());
  test(bulk_asynchronous_executor());
  test(bulk_continuation_executor());

  test(not_a_bulk_synchronous_executor());
  test(not_a_bulk_asynchronous_executor());
  test(not_a_bulk_continuation_executor());

  test(complete_bulk_executor());

  test2(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

