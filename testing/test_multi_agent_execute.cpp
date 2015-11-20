#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <mutex>

#include "test_executors.hpp"

template<class Executor>
void test()
{
  using executor_type = Executor;

  {
    // execute returning user-specified container
    
    executor_type exec;

    size_t n = 100;

    std::vector<int> result = agency::new_executor_traits<executor_type>::execute(exec, [](size_t idx)
    {
      return idx;
    },
    [](size_t n)
    {
      return std::vector<int>(n);
    },
    n);

    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 0);

    assert(std::equal(ref.begin(), ref.end(), result.begin()));
    assert(exec.valid());
  }

  {
    // execute returning default container
    
    executor_type exec;

    size_t n = 100;

    auto result = agency::new_executor_traits<executor_type>::execute(exec, [](size_t idx)
    {
      return idx;
    },
    n);

    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 0);

    assert(std::equal(ref.begin(), ref.end(), result.begin()));
    assert(exec.valid());
  }

  {
    // execute returning void
    
    executor_type exec;

    size_t n = 100;

    int increment_me = 0;
    std::mutex mut;
    agency::new_executor_traits<executor_type>::execute(exec, [&](size_t idx)
    {
      mut.lock();
      increment_me += 13;
      mut.unlock();
    },
    n);

    assert(increment_me == n * 13);
    assert(exec.valid());
  }
}

int main()
{
  using namespace test_executors;

  // a completely empty executor
  test<empty_executor>();

  // single-agent executors
  test<single_agent_execute_executor>();
  test<single_agent_async_execute_executor>();
  test<single_agent_then_execute_executor>();
  test<single_agent_when_all_execute_and_select_executor>();

  // multi-agent when_all_execute_and_select()
  test<multi_agent_when_all_execute_and_select_executor>();
  test<multi_agent_when_all_execute_and_select_with_shared_inits_executor>();

  // multi-agent execute()
  test<multi_agent_execute_with_shared_inits_returning_user_defined_container_executor>();
  test<multi_agent_execute_with_shared_inits_returning_default_container_executor>();
  test<multi_agent_execute_with_shared_inits_returning_void_executor>();

  test<multi_agent_execute_returning_user_defined_container_executor>();
  test<multi_agent_execute_returning_default_container_executor>();
  test<multi_agent_execute_returning_void_executor>();

  // multi-agent async_execute()
  test<multi_agent_async_execute_with_shared_inits_returning_user_defined_container_executor>();
  test<multi_agent_async_execute_with_shared_inits_returning_default_container_executor>();
  test<multi_agent_async_execute_with_shared_inits_returning_void_executor>();

  test<multi_agent_async_execute_returning_user_defined_container_executor>();
  test<multi_agent_async_execute_returning_default_container_executor>();
  test<multi_agent_async_execute_returning_void_executor>();

  // multi-agent then_execute()
  test<multi_agent_then_execute_with_shared_inits_returning_user_defined_container_executor>();
  test<multi_agent_then_execute_with_shared_inits_returning_default_container_executor>();
  test<multi_agent_then_execute_with_shared_inits_returning_void_executor>();

  test<multi_agent_then_execute_returning_user_defined_container_executor>();
  test<multi_agent_then_execute_returning_default_container_executor>();
  test<multi_agent_then_execute_returning_void_executor>();

  std::cout << "OK" << std::endl;

  return 0;
}

