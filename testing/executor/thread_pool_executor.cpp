#include <iostream>
#include <type_traits>
#include <vector>

// XXX use parallel_executor.hpp instead of thread_pool.hpp due to circular #inclusion problems
#include <agency/execution/executor/parallel_executor.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/execution/executor/customization_points.hpp>

int main()
{
  using namespace agency;

  static_assert(is_bulk_continuation_executor<detail::thread_pool_executor>::value,
    "thread_pool_executor should be a bulk continuation executor");

  static_assert(is_bulk_executor<detail::thread_pool_executor>::value,
    "thread_pool_executor should be a bulk executor");

  static_assert(detail::is_detected_exact<parallel_execution_tag, new_executor_execution_category_t, detail::thread_pool_executor>::value,
    "thread_pool_executor should have parallel_execution_tag execution_category");

  static_assert(detail::is_detected_exact<size_t, new_executor_shape_t, detail::thread_pool_executor>::value,
    "thread_pool_executor should have size_t shape_type");

  static_assert(detail::is_detected_exact<size_t, new_executor_index_t, detail::thread_pool_executor>::value,
    "thread_pool_executor should have size_t index_type");

  static_assert(detail::is_detected_exact<std::future<int>, new_executor_future_t, detail::thread_pool_executor, int>::value,
    "thread_pool_executor should have std::future future");

  detail::thread_pool_executor exec;

  {
    // bulk_then_execute() with non-void predecessor
    
    std::future<int> predecessor_fut = agency::make_ready_future<int>(exec, 7);

    size_t shape = 10;
    
    auto f = exec.bulk_then_execute(
      [](size_t idx, int& predecessor, std::vector<int>& results, std::vector<int>& shared_arg)
      {
        results[idx] = predecessor + shared_arg[idx];
      },
      shape,
      predecessor_fut,
      [=]{ return std::vector<int>(shape); },     // results
      [=]{ return std::vector<int>(shape, 13); }  // shared_arg
    );
    
    auto result = f.get();
    
    assert(std::vector<int>(10, 7 + 13) == result);
  }


  {
    // bulk_then_execute() with void predecessor
    
    std::future<void> predecessor_fut = agency::make_ready_future<void>(exec);

    size_t shape = 10;
    
    auto f = exec.bulk_then_execute(
      [](size_t idx, std::vector<int>& results, std::vector<int>& shared_arg)
      {
        results[idx] = shared_arg[idx];
      },
      shape,
      predecessor_fut,
      [=]{ return std::vector<int>(shape); },     // results
      [=]{ return std::vector<int>(shape, 13); }  // shared_arg
    );
    
    auto result = f.get();
    
    assert(std::vector<int>(10, 13) == result);
  }

  std::cout << "OK" << std::endl;

  return 0;
}


