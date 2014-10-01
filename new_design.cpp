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

  bulk_async(seq(2), [](std::sequential_agent &self)
  {
    int i = self.index();

    std::cout << i << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::con" << std::endl << std::endl;

  bulk_async(con(10), [](std::concurrent_agent &self)
  {
    std::cout << "agent " << self.index() << " arriving at barrier" << std::endl;

    self.wait();

    std::cout << "departing barrier" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::par" << std::endl << std::endl;

  bulk_async(par(20), [](std::parallel_agent &self)
  {
    std::cout << "agent " << self.index() << " in par group" << std::endl;
  }).wait();

  std::cout << std::endl;


  std::cout << "Testing std::seq(std::seq)" << std::endl << std::endl;

  std::mutex mut;
  auto singly_nested_f = bulk_async(con(2, seq(3)), [&mut](std::concurrent_group<std::sequential_agent> &self)
  {
    mut.lock();
    std::cout << "Hello world from con(seq) agent (" << self.index() << ", " << self.inner().index() << ")" << std::endl;
    mut.unlock();

    // the first agent in each inner group waits on the outer group 
    if(self.inner().index() == 0)
    {
      mut.lock();
      std::cout << "con(seq) agent " << std::int2(self.index(), self.inner().index()) << " arriving at barrier" << std::endl;
      mut.unlock();

      self.wait();

      mut.lock();
      std::cout << "con(seq) agent (" << self.index() << ", " << self.inner().index() << ") departing barrier" << std::endl;
      mut.unlock();
    }
  });

  singly_nested_f.wait();

  auto doubly_nested_f = bulk_async(seq(2, par(2, seq(3))), [&mut](std::sequential_group<std::parallel_group<std::sequential_agent>> &self)
  {
    mut.lock();
    std::cout << "Hello world from sequential_agent " << self.inner().inner().index() << " of parallel_group " << self.inner().index() << " of sequential_group " << self.index() << std::endl;
    mut.unlock();
  });

  doubly_nested_f.wait();

  return 0;
}

