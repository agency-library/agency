#include <agency/new_executor_traits.hpp>
#include <cassert>
#include <iostream>

#include "test_executors.hpp"

template<class Executor>
void test()
{
  using executor_type = Executor;

  {
    // returning void
    executor_type exec;

    int set_me_to_thirteen = 0;

    agency::new_executor_traits<executor_type>::execute(exec, [&]
    {
      set_me_to_thirteen = 13;
    });

    assert(set_me_to_thirteen == 13);
    assert(exec.valid());
  }

  {
    // returning int
    executor_type exec;

    auto result = agency::new_executor_traits<executor_type>::execute(exec, []
    {
      return 13;
    });

    assert(result == 13);
    assert(exec.valid());
  }
}

int main()
{
  using namespace test_executors;

  test<empty_executor>();
  test<single_agent_when_all_execute_and_select_executor>();
  test<multi_agent_when_all_execute_and_select_executor>();
  test<single_agent_then_execute_executor>();
  test<multi_agent_execute_returning_user_defined_container_executor>();
  test<multi_agent_execute_returning_default_container_executor>();

  std::cout << "OK" << std::endl;

  return 0;
}

