#include <cassert>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <agency/executor_array.hpp>
#include <agency/concurrent_executor.hpp>
#include <agency/flattened_executor.hpp>

int main()
{
  using namespace agency;

  using inner_executor_type = concurrent_executor;

  {
    // test executor_array async_execute()
    using executor_type = executor_array<inner_executor_type>;
    using traits = agency::executor_traits<executor_type>;
    using shape_type = typename traits::shape_type;
    using index_type = typename traits::index_type;

    executor_type exec(2);

    auto shape = exec.make_shape(3,5);

    std::mutex mut;
    auto f = traits::async_execute(exec, [=,&mut](const index_type& idx, int& outer_shared, int& inner_shared)
    {
      mut.lock();
      std::cout << "Hello from agent " << idx << std::endl;
      mut.unlock();

      return 13 + outer_shared + inner_shared;
    },
    [](shape_type shape)
    {
      return traits::container<int>(shape);
    },
    shape,
    []{ return 7; },
    []{ return 42; });

    // sleep for a bit
    mut.lock();
    std::cout << "main thread sleeping for a bit..." << std::endl;
    mut.unlock();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    mut.lock();
    std::cout << "main thread woke up" << std::endl;
    mut.unlock();

    auto results = f.get();

    assert(results.size() == agency::detail::shape_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7 + 42; }));
  }

  {
    // test executor_array then_execute()
    using executor_type = executor_array<inner_executor_type>;
    using traits = agency::executor_traits<executor_type>;
    using shape_type = typename traits::shape_type;
    using index_type = typename traits::index_type;

    executor_type exec(2);

    auto past = traits::make_ready_future<int>(exec,1);

    auto shape = exec.make_shape(3,5);

    std::mutex mut;
    auto f = traits::then_execute(exec, [=,&mut](const index_type& idx, int& past, int& outer_shared, int& inner_shared)
    {
      mut.lock();
      std::cout << "Hello from agent " << idx << std::endl;
      mut.unlock();

      return 13 + past + outer_shared + inner_shared;
    },
    [](shape_type shape)
    {
      return traits::container<int>(shape);
    },
    shape,
    past,
    []{ return 7; },
    []{ return 42; });

    // sleep for a bit
    mut.lock();
    std::cout << "main thread sleeping for a bit..." << std::endl;
    mut.unlock();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    mut.lock();
    std::cout << "main thread woke up" << std::endl;
    mut.unlock();

    auto results = f.get();

    assert(results.size() == agency::detail::shape_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 1 + 7 + 42; }));
  }

  {
    // test executor_array then_execute() returning void
    using executor_type = executor_array<inner_executor_type>;
    using traits = agency::executor_traits<executor_type>;
    using shape_type = typename traits::shape_type;
    using index_type = typename traits::index_type;

    executor_type exec(2);

    auto past = traits::make_ready_future<int>(exec,1);

    auto shape = exec.make_shape(3,5);

    std::atomic<int> result{0};

    std::mutex mut;
    auto f = traits::then_execute(exec, [=,&mut,&result](const index_type& idx, int& past, int& outer_shared, int& inner_shared)
    {
      mut.lock();
      std::cout << "Hello from agent " << idx << std::endl;
      mut.unlock();

      result += 13 + past + outer_shared + inner_shared;
    },
    shape,
    past,
    []{ return 7; },
    []{ return 42; });

    // sleep for a bit
    mut.lock();
    std::cout << "main thread sleeping for a bit..." << std::endl;
    mut.unlock();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    mut.lock();
    std::cout << "main thread woke up" << std::endl;
    mut.unlock();

    f.wait();

    assert(result == agency::detail::shape_size(shape) * (13 + 1 + 7 + 42));
  }

  {
    // test flattened executor_array
    using executor_array_type = executor_array<inner_executor_type>;
    using executor_type = flattened_executor<executor_array_type>;

    using traits = agency::executor_traits<executor_type>;
    using shape_type = typename traits::shape_type;
    using index_type = typename traits::index_type;

    executor_array_type exec_array(2);
    executor_type exec{exec_array};

    shape_type shape = 10;

    auto f = traits::async_execute(exec, [](const index_type& idx, int& shared)
    {
      return 13 + shared;
    },
    [](shape_type shape)
    {
      return traits::container<int>(shape);
    },
    shape,
    []{ return 7; });

    auto results = f.get();

    assert(results.size() == agency::detail::shape_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7; }));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

