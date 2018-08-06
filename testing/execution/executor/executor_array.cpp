#include <iostream>
#include <type_traits>
#include <vector>
#include <cassert>

#include <agency/execution/executor/executor_array.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/tuple.hpp>
#include "test_executors.hpp"

template<class OuterExecutor, class InnerExecutor>
void test(OuterExecutor, InnerExecutor inner_exec)
{
  using namespace agency;

  using executor_array_type = agency::executor_array<InnerExecutor,OuterExecutor>;

  static_assert(detail::is_bulk_then_executor<executor_array_type>::value,
    "executor_array should be a bulk then executor");

  using expected_bulk_guarantee_type = bulk_guarantee_t::scoped_t<
    decltype(bulk_guarantee_t::static_query<OuterExecutor>()),
    decltype(bulk_guarantee_t::static_query<InnerExecutor>())
  >;

  static_assert(bulk_guarantee_t::static_query<executor_array_type>() == expected_bulk_guarantee_type(),
    "scoped_executor should have expected_bulk_guarantee_type");

  static_assert(detail::is_detected_exact<detail::shape_tuple<size_t,size_t>, executor_shape_t, executor_array_type>::value,
    "executor_array should have detail::shape_tuple<size_t,size_t> shape_type");

  static_assert(detail::is_detected_exact<detail::index_tuple<size_t,size_t>, executor_index_t, executor_array_type>::value,
    "executor_array should have detail::index_tuple<size_t,size_t> index_type");

  static_assert(detail::is_detected_exact<executor_future_t<OuterExecutor,int>, executor_future_t, executor_array_type, int>::value,
    "executor_array should have the same future type as OuterExecutor");

  static_assert(detail::is_detected_exact<executor_allocator_t<OuterExecutor,int>, executor_allocator_t, executor_array_type, int>::value,
    "executor_array should have the same allocator type as OuterExecutor");

  executor_array_type exec(10, inner_exec);

  using shape_type  = executor_shape_t<executor_array_type>;
  using index_type  = executor_index_t<executor_array_type>;
  using result_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<executor_array_type, int>>;

  {
    // test .bulk_then_execute() with non-void predecessor

    shape_type shape(10,10);
    std::future<int> predecessor_fut = make_ready_future<int>(exec, 7);

    auto f = exec.bulk_then_execute(
      [=](index_type idx, int& predecessor, result_type& results, std::vector<int>& outer_shared_arg, std::vector<int>& inner_shared_arg)
      {
        auto outer_idx = agency::get<0>(idx);
        auto inner_idx = agency::get<1>(idx);
        results[idx] = predecessor + outer_shared_arg[outer_idx] + inner_shared_arg[inner_idx];
      },
      shape,
      predecessor_fut,
      [=]{ return result_type(shape); },                          // results
      [=]{ return std::vector<int>(agency::get<0>(shape), 13); }, // outer_shared_arg
      [=]{ return std::vector<int>(agency::get<1>(shape), 42); }  // inner_shared_arg
    );

    auto result = f.get();

    assert(result_type(shape, 7 + 13 + 42) == result);
  }


  {
    // test .bulk_then_execute() with void predecessor

    shape_type shape(10,10);
    std::future<void> predecessor_fut = make_ready_future<void>(exec);

    auto f = exec.bulk_then_execute(
      [=](index_type idx, result_type& results, std::vector<int>& outer_shared_arg, std::vector<int>& inner_shared_arg)
      {
        auto outer_idx = agency::get<0>(idx);
        auto inner_idx = agency::get<1>(idx);
        results[idx] = outer_shared_arg[outer_idx] + inner_shared_arg[inner_idx];
      },
      shape,
      predecessor_fut,
      [=]{ return result_type(shape); },                          // results
      [=]{ return std::vector<int>(agency::get<0>(shape), 13); }, // outer_shared_arg
      [=]{ return std::vector<int>(agency::get<1>(shape), 42); }  // inner_shared_arg
    );

    auto result = f.get();

    assert(result_type(shape, 13 + 42) == result);
  }
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


