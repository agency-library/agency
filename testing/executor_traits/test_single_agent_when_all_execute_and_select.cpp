#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <iostream>
#include <cassert>

#include "test_executors.hpp"

struct move_only
{
  std::future<void> f;

  void operator()(int &x)
  {
    x = 13;
  }

  void operator()(){}
};

template<class Executor>
void test()
{
  using executor_type = Executor;
  
  {
    size_t n = 1;
    
    auto int_ready   = agency::detail::make_ready_future(0);
    auto void_ready  = agency::detail::make_ready_future();
    
    auto futures = std::make_tuple(std::move(void_ready), std::move(int_ready));
    
    std::mutex mut;
    executor_type exec;
    std::future<int> fut = agency::executor_traits<executor_type>::template when_all_execute_and_select<1>(exec, [&mut](int& x)
    {
      mut.lock();
      x += 1;
      mut.unlock();
    },
    std::move(futures));
    
    auto got = fut.get();
    
    assert(got == n);
    assert(exec.valid());
  }

  {
    // test move-only function with argument
    auto int_ready = agency::detail::make_ready_future(0);
    
    auto futures = std::make_tuple(std::move(int_ready));
    
    std::mutex mut;
    executor_type exec;
    std::future<int> fut = agency::executor_traits<executor_type>::template when_all_execute_and_select<0>(exec, move_only(), std::move(futures));
    
    auto got = fut.get();
    
    assert(got == 13);
    assert(exec.valid());
  }

  {
    // test move-only function without argument
    auto void_ready = agency::detail::make_ready_future();
    
    auto futures = std::make_tuple(std::move(void_ready));
    
    std::mutex mut;
    executor_type exec;
    std::future<void> fut = agency::executor_traits<executor_type>::template when_all_execute_and_select(exec, move_only(), std::move(futures));
    
    fut.wait();
    
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

