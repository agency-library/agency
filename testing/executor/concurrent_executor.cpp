#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/execution/executor/concurrent_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits.hpp>

int main()
{
  using namespace agency::detail::new_executor_traits_detail;

  static_assert(is_bulk_continuation_executor<agency::concurrent_executor>::value,
    "concurrent_executor should be a bulk continuation executor");

  static_assert(is_bulk_executor<agency::concurrent_executor>::value,
    "concurrent_executor should be a bulk executor");

  static_assert(agency::detail::is_detected_exact<size_t, executor_shape_t, agency::concurrent_executor>::value,
    "concurrent_executor should have size_t shape_type");

  static_assert(agency::detail::is_detected_exact<size_t, executor_index_t, agency::concurrent_executor>::value,
    "concurrent_executor should have size_t index_type");

  static_assert(agency::detail::is_detected_exact<std::future<int>, executor_future_t, agency::concurrent_executor, int>::value,
    "concurrent_executor should have std::future future");

  agency::concurrent_executor exec;

  std::future<int> fut = agency::detail::make_ready_future<int>(7);

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

