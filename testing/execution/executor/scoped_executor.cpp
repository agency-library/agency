#include <iostream>
#include <type_traits>
#include <vector>
#include <cassert>

#include <agency/execution/executor/scoped_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/customization_points.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/tuple.hpp>
#include "test_executors.hpp"

template<class OuterExecutor, class InnerExecutor>
void test(OuterExecutor outer_exec, InnerExecutor inner_exec)
{
  using namespace agency;

  using scoped_executor_type = scoped_executor<OuterExecutor,InnerExecutor>;

  static_assert(detail::is_bulk_then_executor<scoped_executor_type>::value,
    "scoped_executor should be a bulk then executor");

  using expected_bulk_guarantee_type = bulk_guarantee_t::scoped_t<
    decltype(bulk_guarantee_t::static_query<OuterExecutor>()),
    decltype(bulk_guarantee_t::static_query<InnerExecutor>())
  >;

  static_assert(bulk_guarantee_t::static_query<scoped_executor_type>() == expected_bulk_guarantee_type(),
    "scoped_executor should have expected_bulk_guarantee_type");

  static_assert(detail::is_detected_exact<detail::shape_tuple<size_t,size_t>, executor_shape_t, scoped_executor_type>::value,
    "scoped_executor should have detail::shape_tuple<size_t,size_t> shape_type");

  static_assert(detail::is_detected_exact<detail::index_tuple<size_t,size_t>, executor_index_t, scoped_executor_type>::value,
    "scoped_executor should have detail::index_tuple<size_t,size_t> index_type");

  static_assert(detail::is_detected_exact<executor_future_t<OuterExecutor,int>, executor_future_t, scoped_executor_type, int>::value,
    "scoped_executor should have the same future type as OuterExecutor");

  const size_t outer_depth = executor_execution_depth<OuterExecutor>::value;
  const size_t inner_depth = executor_execution_depth<InnerExecutor>::value;

  static_assert(executor_execution_depth<scoped_executor_type>::value == outer_depth + inner_depth,
    "scoped_executor should have execution_depth == outer_depth + inner_depth");

  scoped_executor_type exec(outer_exec,inner_exec);

  std::future<int> fut = make_ready_future<int>(exec, 7);

  using shape_type = executor_shape_t<scoped_executor_type>;
  shape_type shape(10,10);

  using index_type = executor_index_t<scoped_executor_type>;

  auto f = exec.bulk_then_execute(
    [=](index_type idx, int& past_arg, std::vector<int>& results, std::vector<int>& outer_shared_arg, std::vector<int>& inner_shared_arg)
    {
      auto rank = agency::get<0>(idx) * agency::get<1>(shape) + agency::get<1>(idx);
      auto outer_idx = agency::get<0>(idx);
      auto inner_idx = agency::get<1>(idx);
      results[rank] = past_arg + outer_shared_arg[outer_idx] + inner_shared_arg[inner_idx];
    },
    shape,
    fut,
    [=]{ return std::vector<int>(detail::shape_cast<int>(shape)); }, // results
    [=]{ return std::vector<int>(agency::get<0>(shape), 13); },      // outer_shared_arg
    [=]{ return std::vector<int>(agency::get<1>(shape), 42); }       // inner_shared_arg
  );

  auto result = f.get();

  assert(std::vector<int>(10 * 10, 7 + 13 + 42) == result);
}

int main()
{
  // XXX nomerge
  // XXX why are these tests commented out?
  //     see if we can reenable before merging
  test(bulk_then_executor(), bulk_then_executor());
  //test(bulk_then_executor(), bulk_synchronous_executor());
  //test(bulk_then_executor(), bulk_asynchronous_executor());

  //test(bulk_synchronous_executor(), bulk_then_executor());
  //test(bulk_synchronous_executor(), bulk_synchronous_executor());
  //test(bulk_synchronous_executor(), bulk_asynchronous_executor());

  //test(bulk_asynchronous_executor(), bulk_then_executor());
  //test(bulk_asynchronous_executor(), bulk_synchronous_executor());
  //test(bulk_asynchronous_executor(), bulk_asynchronous_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

