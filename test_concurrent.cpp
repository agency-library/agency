#include <iostream>
#include <execution_policy>

int main()
{
  std::bulk_async(std::con(10), [](std::concurrent_agent<> &g)
  {
    std::cout << "agent " << g.child().index() << " arriving at barrier" << std::endl;

    g.wait();

    std::cout << "departing barrier" << std::endl;
  }).wait();

  return 0;
}

