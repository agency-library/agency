#include <agency/agency.hpp>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>

int main()
{
  using namespace agency;

  using inner_executor_type = concurrent_executor;

  {
    // test bulk_twoway_execute()
    using executor_type = executor_array<inner_executor_type>;
    using shape_type = executor_shape_t<executor_type>;
    using index_type = executor_index_t<executor_type>;
    using allocator_type = executor_allocator_t<executor_type, int>;
    using int_container = bulk_result<int, shape_type, allocator_type>;

    executor_type exec(2);

    shape_type shape = exec.make_shape(3,5);

    std::mutex mut;
    auto f = agency::detail::bulk_twoway_execute(exec, [=,&mut](const index_type& idx, int_container& results, int& outer_shared, int& inner_shared)
    {
      mut.lock();
      std::cout << "Hello from agent " << idx << std::endl;
      mut.unlock();

      results[idx] = 13 + outer_shared + inner_shared;
    },
    shape,
    [=]{ return int_container(shape); },
    []{ return 7; },
    []{ return 42; }
    );

    // sleep for a bit
    mut.lock();
    std::cout << "main thread sleeping for a bit..." << std::endl;
    mut.unlock();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    mut.lock();
    std::cout << "main thread woke up" << std::endl;
    mut.unlock();

    auto results = f.get();

    assert(results.size() == agency::detail::index_space_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7 + 42; }));
  }

  {
    // test bulk_then_execute()
    using executor_type = executor_array<inner_executor_type>;
    using shape_type = executor_shape_t<executor_type>;
    using index_type = executor_index_t<executor_type>;
    using allocator_type = executor_allocator_t<executor_type, int>;
    using int_container = bulk_result<int, shape_type, allocator_type>;

    executor_type exec(2);

    auto past = agency::make_ready_future<int>(exec,1);

    shape_type shape = exec.make_shape(3,5);

    std::mutex mut;
    auto f = agency::detail::bulk_then_execute(exec, [=,&mut](const index_type& idx, int& past, int_container& results, int& outer_shared, int& inner_shared)
    {
      mut.lock();
      std::cout << "Hello from agent " << idx << std::endl;
      mut.unlock();

      results[idx] = 13 + past + outer_shared + inner_shared;
    },
    shape,
    past,
    [=]{ return int_container(shape); },
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

    assert(results.size() == agency::detail::index_space_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 1 + 7 + 42; }));
  }

  {
    // test flattened executor_array
    using executor_array_type = executor_array<inner_executor_type>;
    using executor_type = flattened_executor<executor_array_type>;

    using shape_type = executor_shape_t<executor_type>;
    using index_type = executor_index_t<executor_type>;
    using allocator_type = executor_allocator_t<executor_type, int>;
    using int_container = bulk_result<int, shape_type, allocator_type>;

    executor_array_type exec_array(2);
    executor_type exec{exec_array};

    shape_type shape = 10;

    auto f = agency::detail::bulk_twoway_execute(exec, [](const index_type& idx, int_container& results, int& shared)
    {
      results[idx] = 13 + shared;
    },
    shape,
    [=]{ return int_container(shape); },
    []{ return 7; }
    );

    auto results = f.get();

    assert(results.size() == agency::detail::index_space_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7; }));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

