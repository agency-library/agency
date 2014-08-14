#include <iostream>
#include "processor.hpp"

int main()
{
  std::cout << "main running on CPU " << this_cpu << std::endl;

  async(this_cpu, []
  {
    std::cout << "thread " << std::this_thread::get_id() << " on CPU " << this_cpu << std::endl;
  }).wait();

  async(cpu_id(3), []
  {
    std::cout << "thread " << std::this_thread::get_id() << " on CPU " << this_cpu << std::endl;
  }).wait();

  async(this_processor, []
  {
    std::cout << "thread " << std::this_thread::get_id() << " on processor " << this_processor << " of type " << this_processor.type().name() << std::endl;
  }).wait();

  async(this_cpu, [](int x)
  {
    std::cout << "CPU " << this_cpu << " received " << x << std::endl;
  },
  13).wait();

  async(this_processor, [](int x)
  {
    std::cout << "processor " << this_processor << " received " << x << std::endl;
  },
  13).wait();

  return 0;
}

