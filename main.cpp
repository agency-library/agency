#include <iostream>
#include <execution_policy>

std::mutex mut;

int main()
{
  using std::seq;
  using std::par;
  using std::con;

  auto seq_f = bulk_async(seq(5), [](std::sequential_agent &self)
  {
    std::cout << "Hello world from sequential_agent " << self.index() << std::endl;
  });

  auto par_f = bulk_async(par(5), [](std::parallel_agent &self)
  {
    mut.lock();
    std::cout << "Hello world from parallel_agent " << self.index() << std::endl;
    mut.unlock();
  });

  auto con_f = bulk_async(con(5), [](std::concurrent_agent &self)
  {
    mut.lock();
    std::cout << "Hello world from concurrent_agent " << self.index() << " arriving at barrier." << std::endl;
    mut.unlock();

    self.wait();

    mut.lock();
    std::cout << "Hello world from concurrent_agent " << self.index() << " departing from barrier." << std::endl;
    mut.unlock();
  });

  seq_f.wait();
  par_f.wait();
  con_f.wait();

  auto singly_nested_f = bulk_async(con(2, seq(3)), [](std::concurrent_group<std::sequential_agent> &self)
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

  auto doubly_nested_f = bulk_async(seq(2, par(2, seq(3))), [](std::sequential_group<std::parallel_group<std::sequential_agent>> &self)
  {
    mut.lock();
    std::cout << "Hello world from sequential_agent " << self.inner().inner().index() << " of parallel_group " << self.inner().outer().index() << " of sequential_group " << self.outer().index() << std::endl;
    mut.unlock();
  });

  doubly_nested_f.wait();

  return 0;
}

