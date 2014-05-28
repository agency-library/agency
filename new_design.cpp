#include <iostream>
#include <execution_policy_new>
#include <mutex>

int main()
{
  std::cout << "Testing std::seq" << std::endl << std::endl;

  std::async(std::seq(2), [](std::sequential_group<> &g)
  {
    int i = g.child().index();

    std::cout << i << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::con" << std::endl << std::endl;

  std::async(std::con(10), [](std::concurrent_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " arriving at barrier" << std::endl;

    g.wait();

    std::cout << "departing barrier" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::par" << std::endl << std::endl;

  std::async(std::par(20), [](std::parallel_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " in par group" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::seq.on()" << std::endl << std::endl;

  auto cpu = std::cpu_id(3);

  std::async(std::seq(10).on(cpu), [](std::sequential_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " on processor " << std::this_processor << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::con.on()" << std::endl << std::endl;

  std::mutex mut;
  std::async(std::con(10).on(cpu), [&mut](std::concurrent_group<> &g)
  {
    mut.lock();
    std::cout << "agent " << g.child().index() << " on processor " << std::this_processor << " arriving at barrier" << std::endl;
    mut.unlock();

    g.wait();

    mut.lock();
    std::cout << "agent " << g.child().index() << " departing barrier" << std::endl;
    mut.unlock();
  }).wait();

  std::cout << std::endl;


  return 0;
}

