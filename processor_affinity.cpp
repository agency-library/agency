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

  return 0;
}

