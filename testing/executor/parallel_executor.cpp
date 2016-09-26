#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/execution/executor/parallel_executor.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/execution/executor/customization_points.hpp>

int main()
{
  using namespace agency;

  static_assert(is_bulk_continuation_executor<parallel_executor>::value,
    "parallel_executor should be a bulk continuation executor");

  static_assert(is_bulk_executor<parallel_executor>::value,
    "parallel_executor should be a bulk executor");

  static_assert(detail::is_detected_exact<parallel_execution_tag, new_executor_execution_category_t, parallel_executor>::value,
    "parallel_executor should have parallel_execution_tag execution_category");

  static_assert(detail::is_detected_exact<size_t, new_executor_shape_t, parallel_executor>::value,
    "parallel_executor should have size_t shape_type");

  static_assert(detail::is_detected_exact<size_t, new_executor_index_t, parallel_executor>::value,
    "parallel_executor should have size_t index_type");

  static_assert(detail::is_detected_exact<std::future<int>, new_executor_future_t, parallel_executor, int>::value,
    "parallel_executor should have std::future future");

  parallel_executor exec;

  std::future<int> fut = agency::make_ready_future<int>(exec, 7);

  size_t shape = 10;
  
  auto f = exec.bulk_then_execute(
    [](size_t idx, int& past_arg, std::vector<int>& results, std::vector<int>& shared_arg)
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

  std::cout << "OK" << std::endl;

  return 0;
}
