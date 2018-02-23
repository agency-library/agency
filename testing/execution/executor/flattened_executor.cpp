#include <iostream>
#include <type_traits>
#include <vector>
#include <cassert>

#include <agency/execution/executor/scoped_executor.hpp>
#include <agency/execution/executor/flattened_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/customization_points.hpp>
#include "test_executors.hpp"

template<class OuterExecutor, class InnerExecutor>
void test(OuterExecutor outer_exec, InnerExecutor inner_exec)
{
  using namespace agency;

  using scoped_executor_type = scoped_executor<OuterExecutor,InnerExecutor>;
  using flattened_executor_type = flattened_executor<scoped_executor_type>;

  static_assert(detail::is_bulk_then_executor<flattened_executor_type>::value,
    "flattened_executor should be a bulk then executor");

  static_assert(detail::is_detected_exact<size_t, executor_shape_t, flattened_executor_type>::value,
    "flattened_executor should have size_t shape_type");

  static_assert(detail::is_detected_exact<size_t, executor_index_t, flattened_executor_type>::value,
    "flattened_executor should have size_t index_type");

  static_assert(detail::is_detected_exact<executor_future_t<OuterExecutor,int>, executor_future_t, flattened_executor_type, int>::value,
    "flattened_executor should have the same future type as OuterExecutor");

  const size_t scoped_depth = executor_execution_depth<scoped_executor_type>::value;

  static_assert(executor_execution_depth<flattened_executor_type>::value == scoped_depth - 1,
    "flattened_executor should have execution_depth == scoped_depth - 1");

  flattened_executor_type exec(scoped_executor_type(outer_exec,inner_exec));

  std::future<int> fut = make_ready_future<int>(exec, 7);

  using shape_type = executor_shape_t<flattened_executor_type>;
  shape_type shape(10);

  using index_type = executor_index_t<flattened_executor_type>;

  auto f = exec.bulk_then_execute(
    [=](index_type idx, int& past_arg, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = past_arg + shared_arg[idx];
    },
    shape,
    fut,
    [=]{ return std::vector<int>(shape); },    // results
    [=]{ return std::vector<int>(shape, 13); } // shared_arg
  );

  auto result = f.get();

  assert(std::vector<int>(shape, 7 + 13) == result);
}

int main()
{
  test(bulk_then_executor(), bulk_then_executor());
  test(bulk_then_executor(), bulk_twoway_executor());

  test(bulk_twoway_executor(), bulk_then_executor());
  test(bulk_twoway_executor(), bulk_twoway_executor());

  std::cout << "OK" << std::endl;

  return 0;
}


