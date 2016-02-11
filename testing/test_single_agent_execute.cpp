#include <agency/executor_traits.hpp>
#include <future>
#include <cassert>
#include <iostream>

#include "test_executors.hpp"

struct move_only
{
  std::future<void> f;

  int operator()()
  {
    return 13;
  }
};

template<class Executor>
void test()
{
  using executor_type = Executor;

  {
    // returning void
    executor_type exec;

    int set_me_to_thirteen = 0;

    agency::executor_traits<executor_type>::execute(exec, [&]
    {
      set_me_to_thirteen = 13;
    });

    assert(set_me_to_thirteen == 13);
    assert(exec.valid());
  }

  {
    // returning int
    executor_type exec;

    auto result = agency::executor_traits<executor_type>::execute(exec, []
    {
      return 13;
    });

    assert(result == 13);
    assert(exec.valid());
  }

  {
    // with move-only functor
    executor_type exec;

    auto result = agency::executor_traits<executor_type>::execute(exec, move_only());

    assert(result == 13);
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

