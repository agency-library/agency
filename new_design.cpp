#include <iostream>
#include <execution_policy>
#include "processor.hpp"
#include <mutex>

int main()
{
  using std::seq;
  using std::con;
  using std::par;

  std::cout << "Testing seq" << std::endl << std::endl;

  bulk_async(seq(2), [](std::sequential_agent<> &g)
  {
    int i = g.index();

    std::cout << i << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::con" << std::endl << std::endl;

  bulk_async(con(10), [](std::concurrent_agent<> &g)
  {
    std::cout << "agent " << g.index() << " arriving at barrier" << std::endl;

    g.wait();

    std::cout << "departing barrier" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::par" << std::endl << std::endl;

  bulk_async(par(20), [](std::parallel_agent<> &g)
  {
    std::cout << "agent " << g.index() << " in par group" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::seq.on()" << std::endl << std::endl;

  auto cpu = cpu_id(3);

  bulk_async(seq(10).on(cpu), [](std::sequential_agent<> &g)
  {
    std::cout << "agent " << g.index() << " on processor " << this_processor << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::con.on()" << std::endl << std::endl;

  std::mutex mut;
  bulk_async(con(10).on(cpu), [&mut](std::concurrent_agent<> &g)
  {
    mut.lock();
    std::cout << "agent " << g.index() << " on processor " << this_processor << " arriving at barrier" << std::endl;
    mut.unlock();

    g.wait();

    mut.lock();
    std::cout << "agent " << g.index() << " departing barrier" << std::endl;
    mut.unlock();
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::seq(std::seq)" << std::endl << std::endl;


  auto singly_nested_f = bulk_async(con(2, seq(3)), [&mut](std::concurrent_agent<std::sequential_agent<>> &g)
  {
    mut.lock();
    std::cout << "Hello world from con(seq) agent (" << g.index() << ", " << g.child().index() << ")" << std::endl;
    mut.unlock();

    // the first agent in each inner group waits on the outer group 
    if(g.child().index() == 0)
    {
      mut.lock();
      std::cout << "con(seq) agent " << std::int2(g.index(), g.child().index()) << " arriving at barrier" << std::endl;
      mut.unlock();

      g.wait();

      mut.lock();
      std::cout << "con(seq) agent (" << g.index() << ", " << g.child().index() << ") departing barrier" << std::endl;
      mut.unlock();
    }
  });

  singly_nested_f.wait();

  auto doubly_nested_f = bulk_async(seq(2, par(2, seq(3))), [&mut](std::sequential_agent<std::parallel_agent<std::sequential_agent<>>> &g)
  {
    mut.lock();
    std::cout << "Hello world from sequential_agent " << g.child().child().index() << " of parallel_agent " << g.child().index() << " of sequential_agent " << g.index() << std::endl;
    mut.unlock();
  });

  doubly_nested_f.wait();

  return 0;
}

