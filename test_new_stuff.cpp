#include <execution_policy>
#include <iostream>

int main()
{
  bulk_async(std::seq, [](std::sequential_agent_new& self)
  {
    std::cout << "self.index(): " << self.index() << std::endl;
  }).wait();

  std::mutex mut;
  bulk_invoke(std::con(10), [&mut](std::concurrent_agent_new& self)
  {
    mut.lock();
    std::cout << "self.index(): " << self.index() << " arriving at barrier" << std::endl;
    mut.unlock();

    self.wait();

    mut.lock();
    std::cout << "self.index(): " << self.index() << " departing barrier" << std::endl;
    mut.unlock();
  });

  bulk_async(std::seq(3, std::seq(2)), [](std::sequential_group_new<std::sequential_agent_new>& self)
  {
    std::cout << "index: (" << self.index() << ", " << self.inner().index() << ")" << std::endl;
  }).wait();

  bulk_invoke(std::seq(4, std::seq(3, std::seq(2))), [](std::sequential_group_new<std::sequential_group_new<std::sequential_agent_new>>& self)
  {
    std::cout << "index: (" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ")" << std::endl;
  });

  bulk_invoke(std::con(2, std::seq(2, std::con(2))), [&mut](std::concurrent_group_new<std::sequential_group_new<std::concurrent_agent_new>>& self)
  {
    // the first agent in the first subgroup waits at the top-level barrier
    if(self.inner().index() == 0 && self.inner().inner().index() == 0)
    {
      mut.lock();
      std::cout << "(" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") arriving at top-level barrier" << std::endl;
      mut.unlock();

      self.wait();

      mut.lock();
      std::cout << "(" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") departing top-level barrier" << std::endl;
      mut.unlock();
    }

    // every agent waits at the inner most barrier
    mut.lock();
    std::cout << "  (" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") arriving at bottom-level barrier" << std::endl;
    mut.unlock();

    self.inner().inner().wait();

    mut.lock();
    std::cout << "  (" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") departing bottom-level barrier" << std::endl;
    mut.unlock();
  });

  return 0;
}

