#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/execution/executor/executor_array.hpp>
#include "executor_traits/test_executors.hpp"

template<class OuterExecutor, class InnerExecutor>
void test(OuterExecutor outer_exec, InnerExecutor inner_exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  using executor_array_type = agency::executor_array<OuterExecutor,InnerExecutor>;

  static_assert(is_bulk_continuation_executor<executor_array_type>::value,
    "executor_array should be a bulk continuation executor");

  static_assert(agency::detail::is_detected_exact<agency::detail::tuple<size_t,size_t>, agency::new_executor_shape_t, executor_array_type>::value,
    "executor_array should have detail::tuple<size_t,size_t> shape_type");

  static_assert(agency::detail::is_detected_exact<agency::detail::index_tuple<size_t,size_t>, agency::new_executor_index_t, executor_array_type>::value,
    "executor_array should have detail::index_tuple<size_t,size_t> index_type");

  static_assert(agency::detail::is_detected_exact<executor_future_t<OuterExecutor,int>, executor_future_t, executor_array_type, int>::value,
    "executor_array should have the same future type as OuterExecutor");

  executor_array_type exec(10, inner_exec);

  std::future<int> fut = agency::executor_traits<executor_array_type>::template make_ready_future<int>(exec, 7);

  using shape_type = agency::new_executor_shape_t<executor_array_type>;
  shape_type shape(10,10);

  using index_type = agency::new_executor_index_t<executor_array_type>;

  auto f = exec.bulk_then_execute(
    [=](index_type idx, int& past_arg, std::vector<int>& results, std::vector<int>& outer_shared_arg, std::vector<int>& inner_shared_arg)
    {
      auto rank = agency::detail::get<0>(idx) * agency::detail::get<1>(shape) + agency::detail::get<1>(idx);
      auto outer_idx = agency::detail::get<0>(idx);
      auto inner_idx = agency::detail::get<1>(idx);
      results[rank] = past_arg + outer_shared_arg[outer_idx] + inner_shared_arg[inner_idx];
    },
    shape,
    fut,
    [=]{ return std::vector<int>(agency::detail::shape_cast<int>(shape)); }, // results
    [=]{ return std::vector<int>(agency::detail::get<0>(shape), 13); },      // outer_shared_arg
    [=]{ return std::vector<int>(agency::detail::get<1>(shape), 42); }       // inner_shared_arg
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


