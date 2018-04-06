#include <iostream>
#include <type_traits>
#include <vector>
#include <cassert>
#include <numeric>

#include <agency/execution/executor/experimental/unrolling_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>

int main()
{
  using namespace agency;

  static const size_t unroll_factor = 10;
  using executor_type = experimental::unrolling_executor<unroll_factor>;

  static_assert(detail::is_bulk_twoway_executor<executor_type>::value,
    "unrolling_executor should be a bulk twoway executor");

  static_assert(bulk_guarantee_t::static_query<executor_type>() == bulk_guarantee_t::sequenced_t(),
    "unrolling should have sequenced static bulk guarantee");

  static_assert(detail::is_detected_exact<size_t, executor_shape_t, executor_type>::value,
    "unrolling_executor should have size_t shape_type");

  static_assert(detail::is_detected_exact<size_t, executor_index_t, executor_type>::value,
    "unrolling_executor should have size_t index_type");

  static_assert(detail::is_detected_exact<always_ready_future<int>, executor_future_t, executor_type, int>::value,
    "unrolling_executor should have agency::always_ready_future future");

  static_assert(executor_execution_depth<executor_type>::value == 1,
    "unrolling_executor should have execution_depth == 1");

  executor_type exec;

  std::vector<size_t> shapes = {0, 1, 3, unroll_factor - 1, unroll_factor, unroll_factor + 1, 2 * unroll_factor - 1, 2 * unroll_factor, 100 * unroll_factor, 10000};
  
  for(auto shape : shapes)
  {
    auto result_future = exec.bulk_twoway_execute(
      [](size_t idx, std::vector<int>& results, std::vector<int>& shared_arg)
      {
        results[idx] = idx + shared_arg[idx];
      },
      shape,
      [=]{ return std::vector<int>(shape); },     // results
      [=]{ return std::vector<int>(shape, 13); }  // shared_arg
    );

    auto result = result_future.get();
    
    std::vector<int> reference(shape);
    std::iota(reference.begin(), reference.end(), 13);
    assert(reference == result);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

