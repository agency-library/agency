#include <iostream>
#include <type_traits>
#include <vector>
#include <cassert>

#include <agency/execution/executor/unsequenced_executor.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>

int main()
{
  using namespace agency;

  static_assert(is_bulk_synchronous_executor<unsequenced_executor>::value,
    "unsequenced_executor should be a bulk synchronous executor");

  static_assert(is_bulk_executor<unsequenced_executor>::value,
    "unsequenced_executor should be a bulk executor");

  static_assert(detail::is_detected_exact<unsequenced_execution_tag, executor_execution_category_t, unsequenced_executor>::value,
    "unsequenced_executor should have unsequenced_execution_tag execution_category");

  static_assert(detail::is_detected_exact<size_t, executor_shape_t, unsequenced_executor>::value,
    "unsequenced_executor should have size_t shape_type");

  static_assert(detail::is_detected_exact<size_t, executor_index_t, unsequenced_executor>::value,
    "unsequenced_executor should have size_t index_type");

  static_assert(detail::is_detected_exact<std::future<int>, executor_future_t, unsequenced_executor, int>::value,
    "unsequenced_executor should have std::future furture");

  static_assert(executor_execution_depth<unsequenced_executor>::value == 1,
    "unsequenced_executor should have execution_depth == 1");

  unsequenced_executor exec;

  size_t shape = 10;
  
  auto result = exec.bulk_execute(
    [](size_t idx, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = shared_arg[idx];
    },
    shape,
    [=]{ return std::vector<int>(shape); },     // results
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  assert(std::vector<int>(10, 13) == result);

  std::cout << "OK" << std::endl;

  return 0;
}

