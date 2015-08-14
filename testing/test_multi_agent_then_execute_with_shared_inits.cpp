#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <mutex>

#include "test_executors.hpp"

template<class Executor>
void test()
{
  using executor_type = Executor;

  {
    // then_execute with shared inits returning user-specified container
    // with int future
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future(13);

    int shared_arg = 0;

    std::mutex mut;
    std::future<std::vector<int>> fut = agency::new_executor_traits<executor_type>::template then_execute<std::vector<int>>(exec, [&](size_t idx, int& past, int& shared_arg)
    {
      mut.lock();
      ++shared_arg;
      mut.unlock();
      return past;
    },
    n,
    past,
    [&]
    {
      return std::ref(shared_arg);
    });

    auto got = fut.get();

    assert(got == std::vector<int>(n, 13));
    assert(shared_arg == n);
    assert(exec.valid());
  }

  {
    // then_execute with shared inits returning user-specified container
    // with void future
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future();

    int shared_arg = 0;

    std::mutex mut;
    std::future<std::vector<int>> fut = agency::new_executor_traits<executor_type>::template then_execute<std::vector<int>>(exec, [&](size_t idx, int& shared_arg)
    {
      mut.lock();
      ++shared_arg;
      mut.unlock();
      return 13;
    },
    n,
    past,
    [&]
    {
      return std::ref(shared_arg);
    });

    auto got = fut.get();

    assert(got == std::vector<int>(n, 13));
    assert(shared_arg == n);
    assert(exec.valid());
  }

  {
    // then_execute returning default container
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future(13);

    int shared_arg = 0;

    std::mutex mut;
    auto fut = agency::new_executor_traits<executor_type>::then_execute(exec, [&](size_t idx, int& past, int& shared_arg)
    {
      mut.lock();
      ++shared_arg;
      mut.unlock();
      return past;
    },
    n,
    past,
    [&]
    {
      return std::ref(shared_arg);
    });

    auto result = fut.get();

    std::vector<int> ref(n, 13);
    assert(std::equal(ref.begin(), ref.end(), result.begin()));
    assert(shared_arg == n);
    assert(exec.valid());
  }

  {
    // then_execute with shared inits returning default container
    // with void future
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future();

    int shared_arg = 0;

    std::mutex mut;
    auto fut = agency::new_executor_traits<executor_type>::template then_execute(exec, [&](size_t idx, int& shared_arg)
    {
      mut.lock();
      ++shared_arg;
      mut.unlock();
      return 13;
    },
    n,
    past,
    [&]
    {
      return std::ref(shared_arg);
    });

    auto got = fut.get();

    assert(got == std::vector<int>(n, 13));
    assert(shared_arg == n);
    assert(exec.valid());
  }

  {
    // then_execute with shared inits returning void
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future(13);

    int shared_arg = 0;

    int increment_me = 0;
    std::mutex mut;
    auto fut = agency::new_executor_traits<executor_type>::then_execute(exec, [&](size_t idx, int& past, int& shared_arg)
    {
      mut.lock();
      increment_me += past;
      ++shared_arg;
      mut.unlock();
    },
    n,
    past,
    [&]
    {
      return std::ref(shared_arg);
    });

    fut.wait();

    assert(increment_me == n * 13);
    assert(shared_arg == n);
    assert(exec.valid());
  }

  {
    // then_execute with shared inits returning void
    // with void future
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future();

    int shared_arg = 0;

    int increment_me = 0;
    std::mutex mut;
    auto fut = agency::new_executor_traits<executor_type>::then_execute(exec, [&](size_t idx, int& shared_arg)
    {
      mut.lock();
      increment_me += 1;
      ++shared_arg;
      mut.unlock();
    },
    n,
    past,
    [&]
    {
      return std::ref(shared_arg);
    });

    fut.wait();

    assert(increment_me == n);
    assert(shared_arg == n);
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

