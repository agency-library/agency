#include <agency/new_executor_traits.hpp>
#include <cassert>
#include <iostream>

#include "test_executors.hpp"

template<class Executor>
void test()
{
  using executor_type = Executor;

  {
    // void -> void
    executor_type exec;

    auto void_future = agency::when_all();

    int set_me_to_thirteen = 0;

    auto f = agency::new_executor_traits<executor_type>::then_execute(exec, [&]
    {
      set_me_to_thirteen = 13;
    },
    void_future);

    f.wait();

    assert(set_me_to_thirteen == 13);
    assert(exec.valid());
  }

  {
    // void -> int
    executor_type exec;

    auto void_future = agency::when_all();

    auto f = agency::new_executor_traits<executor_type>::then_execute(exec, []
    {
      return 13;
    },
    void_future);

    assert(f.get() == 13);
    assert(exec.valid());
  }

  {
    // int -> void
    executor_type exec;

    auto int_future = agency::new_executor_traits<executor_type>::template make_ready_future<int>(exec, 13);

    int set_me_to_thirteen = 0;

    auto f = agency::new_executor_traits<executor_type>::then_execute(exec, [&](int& x)
    {
      set_me_to_thirteen = x;
    },
    int_future);

    f.wait();

    assert(set_me_to_thirteen == 13);
    assert(exec.valid());
  }

  {
    // int -> float
    executor_type exec;

    auto int_future = agency::new_executor_traits<executor_type>::template make_ready_future<int>(exec, 13);

    auto f = agency::new_executor_traits<executor_type>::then_execute(exec, [](int &x)
    {
      return float(x) + 1.f;
    },
    int_future);

    assert(f.get() == 14.f);
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
  test<multi_agent_execute_returning_void_executor>();

  test<multi_agent_async_execute_returning_user_defined_container_executor>();
  test<multi_agent_async_execute_returning_default_container_executor>();
  test<multi_agent_async_execute_returning_void_executor>();

  std::cout << "OK" << std::endl;

  return 0;
}

