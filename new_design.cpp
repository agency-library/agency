#include <iostream>
#include <execution_policy_new>

int main()
{
  std::sequential_execution_policy exec = std::seq(2);

  std::async(exec, [](std::sequential_group<> &g)
  {
    int i = g.child().index();

    std::cout << i << std::endl;
  }).wait();

  std::async(std::con(10), [](std::concurrent_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " arriving at barrier" << std::endl;

    g.wait();

    std::cout << "departing barrier" << std::endl;
  }).wait();


  std::async(std::par(20), [](std::parallel_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " in par group" << std::endl;
  }).wait();


  return 0;
}

