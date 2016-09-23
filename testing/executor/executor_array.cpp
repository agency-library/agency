#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/execution/executor/executor_array.hpp>
#include "test_executors.hpp"

template<class OuterExecutor, class InnerExecutor>
void test(OuterExecutor outer_exec, InnerExecutor inner_exec)
{
  using namespace agency;

  using executor_array_type = agency::executor_array<OuterExecutor,InnerExecutor>;

  static_assert(is_bulk_continuation_executor<executor_array_type>::value,
    "executor_array should be a bulk continuation executor");

  using expected_category = scoped_execution_tag<new_executor_execution_category_t<OuterExecutor>, new_executor_execution_category_t<InnerExecutor>>;

  static_assert(detail::is_detected_exact<expected_category, new_executor_execution_category_t, executor_array_type>::value,
    "scoped_executor should have expected_category execution_category");

  static_assert(detail::is_detected_exact<detail::tuple<size_t,size_t>, new_executor_shape_t, executor_array_type>::value,
    "executor_array should have detail::tuple<size_t,size_t> shape_type");

  static_assert(detail::is_detected_exact<detail::index_tuple<size_t,size_t>, new_executor_index_t, executor_array_type>::value,
    "executor_array should have detail::index_tuple<size_t,size_t> index_type");

  static_assert(detail::is_detected_exact<new_executor_future_t<OuterExecutor,int>, new_executor_future_t, executor_array_type, int>::value,
    "executor_array should have the same future type as OuterExecutor");

  executor_array_type exec(10, inner_exec);

  std::future<int> fut = make_ready_future<int>(exec, 7);

  using shape_type = new_executor_shape_t<executor_array_type>;
  shape_type shape(10,10);

  using index_type = new_executor_index_t<executor_array_type>;

  auto f = exec.bulk_then_execute(
    [=](index_type idx, int& past_arg, std::vector<int>& results, std::vector<int>& outer_shared_arg, std::vector<int>& inner_shared_arg)
    {
      auto rank = detail::get<0>(idx) * detail::get<1>(shape) + detail::get<1>(idx);
      auto outer_idx = detail::get<0>(idx);
      auto inner_idx = detail::get<1>(idx);
      results[rank] = past_arg + outer_shared_arg[outer_idx] + inner_shared_arg[inner_idx];
    },
    shape,
    fut,
    [=]{ return std::vector<int>(detail::shape_cast<int>(shape)); }, // results
    [=]{ return std::vector<int>(detail::get<0>(shape), 13); },      // outer_shared_arg
    [=]{ return std::vector<int>(detail::get<1>(shape), 42); }       // inner_shared_arg
  );

  auto result = f.get();

  assert(std::vector<int>(10 * 10, 7 + 13 + 42) == result);
}

int main()
{
  test(bulk_continuation_executor(), bulk_continuation_executor());
  //test(bulk_continuation_executor(), bulk_synchronous_executor());
  //test(bulk_continuation_executor(), bulk_asynchronous_executor());

  //test(bulk_synchronous_executor(), bulk_continuation_executor());
  //test(bulk_synchronous_executor(), bulk_synchronous_executor());
  //test(bulk_synchronous_executor(), bulk_asynchronous_executor());

  //test(bulk_asynchronous_executor(), bulk_continuation_executor());
  //test(bulk_asynchronous_executor(), bulk_synchronous_executor());
  //test(bulk_asynchronous_executor(), bulk_asynchronous_executor());

  std::cout << "OK" << std::endl;

  return 0;
}


