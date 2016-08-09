#include <agency/agency.hpp>
#include <iostream>

std::mutex mut;

int main()
{
  using namespace agency;

  auto seq_f = bulk_async(seq(5), [](sequenced_agent &self)
  {
    std::cout << "Hello world from sequenced_agent " << self.index() << std::endl;
  });

  seq_f.wait();

  auto par_f = bulk_async(par(5), [](parallel_agent &self)
  {
    mut.lock();
    std::cout << "Hello world from parallel_agent " << self.index() << std::endl;
    mut.unlock();
  });

  par_f.wait();

  auto con_f = bulk_async(con(5), [](concurrent_agent &self)
  {
    mut.lock();
    std::cout << "Hello world from concurrent_agent " << self.index() << " arriving at barrier." << std::endl;
    mut.unlock();

    self.wait();

    mut.lock();
    std::cout << "Hello world from concurrent_agent " << self.index() << " departing from barrier." << std::endl;
    mut.unlock();
  });

  con_f.wait();

  auto singly_nested_f = bulk_async(con(2, seq(3)), [](concurrent_group<sequenced_agent> &self)
  {
    mut.lock();
    std::cout << "Hello world from con(seq) agent " << self.index() << std::endl;
    mut.unlock();

    // the first agent in each inner group waits on the outer group 
    if(self.inner().index() == 0)
    {
      mut.lock();
      std::cout << "con(seq) agent " << self.index() << " arriving at barrier" << std::endl;
      mut.unlock();

      self.outer().wait();

      mut.lock();
      std::cout << "con(seq) agent " << self.index() << " departing barrier" << std::endl;
      mut.unlock();
    }
  });

  singly_nested_f.wait();

  auto doubly_nested_f = bulk_async(seq(2, par(2, seq(3))), [](sequenced_group<parallel_group<sequenced_agent>> &self)
  {
    mut.lock();
    std::cout << "Hello world from sequenced_agent " << self.inner().inner().index() << " of parallel_group " << self.inner().outer().index() << " of sequenced_group " << self.outer().index() << std::endl;
    mut.unlock();
  });

  doubly_nested_f.wait();

  return 0;
}

