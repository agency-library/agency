#include <cassert>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <agency/executor_array.hpp>

int main()
{
  using namespace agency;

  using outer_executor_type = this_thread::parallel_executor;
  using inner_executor_type = concurrent_executor;

  using executor_type = executor_array<inner_executor_type, outer_executor_type>;
  using traits = agency::executor_traits<executor_type>;
  using shape_type = typename traits::shape_type;
  using index_type = typename traits::index_type;

  executor_type exec(2);

  auto shape = exec.make_shape(3,5);

  std::mutex mut;
  auto f = exec.async_execute([=,&mut](const index_type& idx, int& outer_shared)
  {
    mut.lock();
    std::cout << "Hello from agent " << idx << std::endl;
    mut.unlock();

    return 13 + outer_shared;
  },
  [](shape_type shape)
  {
    return traits::container<int>(shape);
  },
  shape,
  []{ return 7; });

  // sleep for a bit
  mut.lock();
  std::cout << "main thread sleeping for a bit..." << std::endl;
  mut.unlock();

  std::this_thread::sleep_for(std::chrono::seconds(3));

  mut.lock();
  std::cout << "main thread woke up" << std::endl;
  mut.unlock();

  auto results = f.get();

  assert(results.size() == agency::detail::shape_size(shape));
  assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7; }));

  std::cout << "OK" << std::endl;

  return 0;
}

