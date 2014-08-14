#include <nested_executor>
#include <concurrent_executor>
#include <sequential_executor>
#include <iostream>
#include <thread>

std::mutex mut;

int main()
{
  std::nested_executor<std::concurrent_executor, std::sequential_executor> ex;

  bulk_async(ex, std::make_pair(2, 2), [](std::pair<size_t,size_t> idx)
  {
    mut.lock();
    std::cout << "Hello world from thread " << std::this_thread::get_id() << std::endl;
    std::cout << "Hello world from index " << idx.first << ", " << idx.second << std::endl;
    mut.unlock();
  }).wait();

  return 0;
}

