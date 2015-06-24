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
    // then_execute returning user-specified container
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future(13);

    std::future<std::vector<int>> fut = agency::new_executor_traits<executor_type>::template then_execute<std::vector<int>>(exec, [](size_t idx, int& past)
    {
      return past;
    },
    n,
    past);

    auto got = fut.get();

    assert(got == std::vector<int>(n, 13));
    assert(exec.valid());
  }

  {
    // then_execute returning default container
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future(13);

    auto fut = agency::new_executor_traits<executor_type>::then_execute(exec, [](size_t idx, int& past)
    {
      return past;
    },
    n,
    past);

    auto result = fut.get();

    std::vector<int> ref(n, 13);
    assert(std::equal(ref.begin(), ref.end(), result.begin()));
    assert(exec.valid());
  }

  {
    // then_execute returning void
    
    executor_type exec;

    size_t n = 100;

    auto past = agency::detail::make_ready_future(13);

    int increment_me = 0;
    std::mutex mut;
    auto fut = agency::new_executor_traits<executor_type>::then_execute(exec, [&](size_t idx, int& past)
    {
      mut.lock();
      increment_me += past;
      mut.unlock();
    },
    n,
    past);

    fut.wait();

    assert(increment_me == n * 13);
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

  std::cout << "OK" << std::endl;

  return 0;
}

