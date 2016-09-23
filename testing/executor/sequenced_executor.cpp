#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/execution/executor/sequenced_executor.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>

int main()
{
  using namespace agency;

  static_assert(is_bulk_synchronous_executor<sequenced_executor>::value,
    "sequenced_executor should be a bulk synchronous executor");

  static_assert(is_bulk_executor<sequenced_executor>::value,
    "sequenced_executor should be a bulk executor");

  static_assert(detail::is_detected_exact<sequenced_execution_tag, new_executor_execution_category_t, sequenced_executor>::value,
    "sequenced_executor should have sequenced_execution_tag execution_category");

  static_assert(detail::is_detected_exact<size_t, new_executor_shape_t, sequenced_executor>::value,
    "sequenced_executor should have size_t shape_type");

  static_assert(detail::is_detected_exact<size_t, new_executor_index_t, sequenced_executor>::value,
    "sequenced_executor should have size_t index_type");

  static_assert(detail::is_detected_exact<std::future<int>, new_executor_future_t, sequenced_executor, int>::value,
    "sequenced_executor should have std::future furture");

  sequenced_executor exec;

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

