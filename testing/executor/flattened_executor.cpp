#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/execution/executor/scoped_executor.hpp>
#include <agency/execution/executor/flattened_executor.hpp>
#include "executor_traits/test_executors.hpp"

template<class OuterExecutor, class InnerExecutor>
void test(OuterExecutor outer_exec, InnerExecutor inner_exec)
{
  using scoped_executor_type = agency::scoped_executor<OuterExecutor,InnerExecutor>;
  using flattened_executor_type = agency::flattened_executor<scoped_executor_type>;

  static_assert(agency::is_bulk_continuation_executor<flattened_executor_type>::value,
    "flattened_executor should be a bulk continuation executor");

  static_assert(agency::detail::is_detected_exact<size_t, agency::new_executor_shape_t, flattened_executor_type>::value,
    "flattened_executor should have size_t shape_type");

  static_assert(agency::detail::is_detected_exact<size_t, agency::new_executor_index_t, flattened_executor_type>::value,
    "flattened_executor should have size_t index_type");

  static_assert(agency::detail::is_detected_exact<agency::new_executor_future_t<OuterExecutor,int>, agency::new_executor_future_t, flattened_executor_type, int>::value,
    "flattened_executor should have the same future type as OuterExecutor");

  flattened_executor_type exec(scoped_executor_type(outer_exec,inner_exec));

  std::future<int> fut = agency::executor_traits<flattened_executor_type>::template make_ready_future<int>(exec, 7);

  using shape_type = agency::new_executor_shape_t<flattened_executor_type>;
  shape_type shape(10);

  using index_type = agency::new_executor_index_t<flattened_executor_type>;

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
  test(bulk_continuation_executor(), bulk_continuation_executor());
  test(bulk_continuation_executor(), bulk_synchronous_executor());
  test(bulk_continuation_executor(), bulk_asynchronous_executor());

  test(bulk_synchronous_executor(), bulk_continuation_executor());
  test(bulk_synchronous_executor(), bulk_synchronous_executor());
  test(bulk_synchronous_executor(), bulk_asynchronous_executor());

  test(bulk_asynchronous_executor(), bulk_continuation_executor());
  test(bulk_asynchronous_executor(), bulk_synchronous_executor());
  test(bulk_asynchronous_executor(), bulk_asynchronous_executor());

  std::cout << "OK" << std::endl;

  return 0;
}


