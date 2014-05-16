#include <iostream>
#include <processor>

int main()
{
  std::cout << "main running on CPU " << std::this_cpu << std::endl;

  std::async(std::this_cpu, []
  {
    std::cout << "thread " << std::this_thread::get_id() << " on CPU " << std::this_cpu << std::endl;
  }).wait();

  std::async(std::cpu_id(3), []
  {
    std::cout << "thread " << std::this_thread::get_id() << " on CPU " << std::this_cpu << std::endl;
  }).wait();

  std::async(std::this_processor, []
  {
    std::cout << "thread " << std::this_thread::get_id() << " on processor " << std::this_processor << " of type " << std::this_processor.type().name() << std::endl;
  }).wait();

  std::async(std::this_processor, [](int x)
  {
    std::cout << "processor " << std::this_processor << " received " << x << std::endl;
  },
  13).wait();

  return 0;
}

