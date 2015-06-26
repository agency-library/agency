#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/coordinate.hpp>
#include <iostream>
#include <cassert>

#include "test_executors.hpp"

template<class Executor>
void test()
{
  using executor_type = Executor;

  {
    executor_type exec;

    size_t n = 100;
    int addend = 13;

    auto futures = std::make_tuple(agency::detail::make_ready_future<int>(addend));

    std::atomic<int> counter(n);
    int current_sum = 0;
    int result = 0;

    std::mutex mut;
    std::future<int> fut = agency::new_executor_traits<executor_type>::template when_all_execute_and_select<0>(exec, [&](size_t idx, int& addend, int& current_sum)
    {
      mut.lock();
      current_sum += addend;
      mut.unlock();

      auto prev_counter_value = counter.fetch_sub(1);

      // the last agent stores the current_sum to the result
      if(prev_counter_value == 1)
      {
        result = current_sum;
      }
    },
    n,
    futures,
    current_sum);

    auto got = fut.get();

    assert(got == 13);
    assert(result == addend * n);
    assert(exec.valid());
  }
}

int main()
{
  using namespace test_executors;

  test<empty_executor>();
  test<single_agent_when_all_execute_and_select_executor>();
  test<multi_agent_when_all_execute_and_select_executor>();
  test<multi_agent_when_all_execute_and_select_with_shared_inits_executor>();

  test<when_all_executor>();

  test<multi_agent_execute_returning_user_defined_container_executor>();
  test<multi_agent_execute_returning_default_container_executor>();
  test<multi_agent_execute_returning_void_executor>();

  test<multi_agent_async_execute_returning_user_defined_container_executor>();
  test<multi_agent_async_execute_returning_default_container_executor>();
  test<multi_agent_async_execute_returning_void_executor>();

  std::cout << "OK" << std::endl;
}

