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

  auto fut = agency::detail::make_ready_future<int>(7);

  using shape_type = executor_shape_t<Executor>;
  using index_type = executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = bulk_then_execute(exec,
    [](index_type idx, int& past_arg, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = past_arg + shared_arg[idx];
    },
    shape,
    fut,
    [=]{ return std::vector<int>(shape); },     // results
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  assert(std::vector<int>(10, 7 + 13) == result);
}


template<class TwoLevelExecutor>
void test2(TwoLevelExecutor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  using predecessor_future_type = executor_future_t<TwoLevelExecutor,int>;
  auto predecessor_future = agency::future_traits<predecessor_future_type>::make_ready(7);

  using shape_type = executor_shape_t<TwoLevelExecutor>;
  using index_type = executor_index_t<TwoLevelExecutor>;

  shape_type shape{10,10};

  using container_type = executor_container_t<TwoLevelExecutor, int>;
  
  auto f = bulk_then_execute(exec,
    [] __device__ (index_type idx, int& past_arg, container_type& results, int& outer_shared_arg, int& inner_shared_arg)
    {
      results[idx] = past_arg + outer_shared_arg + inner_shared_arg;
    },
    shape,
    predecessor_future,
    [=] __host__ __device__ { return container_type(shape); }, // results
    [] __host__ __device__ { return 13; },                     // outer_shared_arg
    [] __host__ __device__ { return 42; }                      // inner_shared_arg
  );
  
  auto result = f.get();
  
  assert(container_type(10, 7 + 13 + 42) == result);
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

