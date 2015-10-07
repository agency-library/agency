#include <agency/execution_policy.hpp>
#include <iostream>

int main()
{
  std::cout << "Starting two tasks asynchronously..." << std::endl;

  std::mutex mut;

  // asynchronously create 5 agents to greet us in bulk
  auto f1 = agency::bulk_async(agency::par(5), [&](agency::parallel_agent& self)
  {
    mut.lock();
    std::cout << "Hello, world from agent " << self.index() << " in task 1" << std::endl;
    mut.unlock();
  });

  // asynchronously create 5 agents to greet us in bulk
  auto f2 = agency::bulk_async(agency::par(5), [&](agency::parallel_agent& self)
  {
    mut.lock();
    std::cout << "Hello, world from agent " << self.index() << " in task 2" << std::endl;
    mut.unlock();
  });

  std::cout << "Sleeping before waiting on the tasks..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << "Woke up, waiting for the tasks to complete..." << std::endl;

  // wait for tasks 1 & 2 to complete before continuing
  f1.wait();
  f2.wait();

  std::cout << "OK" << std::endl;

  return 0;
}

