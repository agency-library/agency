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

  bulk_async(seq(2), [](std::sequential_group<> &g)
  {
    int i = g.child().index();

    std::cout << i << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::con" << std::endl << std::endl;

  bulk_async(con(10), [](std::concurrent_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " arriving at barrier" << std::endl;

    g.wait();

    std::cout << "departing barrier" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::par" << std::endl << std::endl;

  bulk_async(par(20), [](std::parallel_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " in par group" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::seq.on()" << std::endl << std::endl;

  auto cpu = cpu_id(3);

  bulk_async(seq(10).on(cpu), [](std::sequential_group<> &g)
  {
    std::cout << "agent " << g.child().index() << " on processor " << this_processor << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::con.on()" << std::endl << std::endl;

  std::mutex mut;
  bulk_async(con(10).on(cpu), [&mut](std::concurrent_group<> &g)
  {
    mut.lock();
    std::cout << "agent " << g.child().index() << " on processor " << this_processor << " arriving at barrier" << std::endl;
    mut.unlock();

    g.wait();

    mut.lock();
    std::cout << "agent " << g.child().index() << " departing barrier" << std::endl;
    mut.unlock();
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::seq(std::seq)" << std::endl << std::endl;


  auto singly_nested_f = bulk_async(con(2, seq(3)), [&mut](std::concurrent_group<std::sequential_group<>> &g)
  {
    mut.lock();
    std::cout << "Hello world from agent " << g.child().child().index() << " in sequential_group " << g.child().index() << " of concurrent_group " << g.index() << " arriving at barrier" << std::endl;
    mut.unlock();

    g.wait();

    mut.lock();
    std::cout << "Hello world from agent " << g.child().child().index() << " in sequential_group " << g.child().index() << " of concurrent_group " << g.index() << " departing from barrier" << std::endl;
    mut.unlock();
  });

  singly_nested_f.wait();

  auto doubly_nested_f = bulk_async(seq(2, par(2, seq(3))), [&mut](std::sequential_group<std::parallel_group<std::sequential_group<>>> &g)
  {
    mut.lock();
    std::cout << "Hello world from agent " << g.child().child().child().index() << " in sequential_group " << g.child().child().index() << " of parallel_group " << g.child().index() << " of sequential_group " << g.index() << std::endl;
    mut.unlock();
  });

  doubly_nested_f.wait();

  return 0;
}

