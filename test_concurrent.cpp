#include <iostream>
#include <execution_policy>

int main()
{
  std::async(std::con(10), [](std::concurrent_group &g)
  {
    std::cout << "agent " << g.child().index() << " arriving at barrier" << std::endl;

    g.wait();

    std::cout << "departing barrier" << std::endl;
  }).wait();

  return 0;
}

