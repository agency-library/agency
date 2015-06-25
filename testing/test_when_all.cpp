#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <iostream>
#include <cassert>

#include "test_executors.hpp"

template<class Executor>
void test()
{
  using executor_type = Executor;
  using traits = agency::new_executor_traits<executor_type>;

  {
    executor_type exec;

    auto void_future1 = traits::when_all(exec);
    auto void_future2 = traits::when_all(exec, void_future1);

    void_future2.get();
  }

  {
    executor_type exec;

    auto int_ready   = traits::template make_ready_future<int>(exec, 13);
    auto float_ready = traits::template make_ready_future<float>(exec, 7.f);

    auto int_float_fut = traits::when_all(exec, int_ready, float_ready);

    auto int_float = int_float_fut.get();

    assert(std::get<0>(int_float) == 13);
    assert(std::get<1>(int_float) == 7.f);
  }

  {
    executor_type exec;

    auto int_ready   = agency::detail::make_ready_future(13);
    auto void_ready  = agency::detail::make_ready_future();
    auto float_ready = agency::detail::make_ready_future(7.f);

    auto int_float_fut = traits::when_all(exec, int_ready, void_ready, float_ready);

    auto int_float = int_float_fut.get();

    assert(std::get<0>(int_float) == 13);
    assert(std::get<1>(int_float) == 7.f);
  }
}

int main()
{
  using namespace test_executors;

  test<empty_executor>();
  test<single_agent_when_all_execute_and_select_executor>();
  test<multi_agent_when_all_execute_and_select_executor>();
  test<single_agent_then_execute_executor>();
  test<when_all_executor>();
  test<multi_agent_execute_returning_user_defined_container_executor>();
  test<multi_agent_execute_returning_default_container_executor>();

  std::cout << "OK" << std::endl;

  return 0;
}

