#include <agency/agency.hpp>
#include <iostream>
#include <mutex>

int main()
{
  using namespace agency;

  std::cout << "Starting predecessor and continuation tasks asynchronously..." << std::endl;

  std::mutex mut;

  // asynchronously create 5 agents to greet us in a predecessor task
  std::future<void> predecessor = bulk_async(par(5), [&](parallel_agent& self)
  {
    mut.lock();
    std::cout << "Hello, world from agent " << self.index() << " in the predecessor task" << std::endl;
    mut.unlock();
  });

  // create a continuation to the predecessor
  std::future<void> continuation = bulk_then(par(5), [&](parallel_agent& self)
  {
    mut.lock();
    std::cout << "Hello, world from agent " << self.index() << " in the continuation" << std::endl;
    mut.unlock();
  },
  predecessor);

  std::cout << "Sleeping before waiting on the continuation..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << "Woke up, waiting for the continuation to complete..." << std::endl;

  // wait for the continuation to complete before continuing
  continuation.wait();

  std::cout << "OK" << std::endl;
  return 0;
}

