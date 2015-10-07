#include <agency/execution_policy.hpp>
#include <mutex>
#include <iostream>
#include <thread>

int main()
{
  int ball = 0;
  std::string names[2] = {"ping", "pong"};
  std::mutex mut;

  agency::bulk_invoke(agency::con(2), [&](agency::concurrent_agent& self)
  {
    auto name = names[self.index()];

    for(int next_state = self.index();
        next_state < 25;
        next_state += 2)
    {
      while(ball != next_state)
      {
        mut.lock();
        std::cout << name << " waiting for return" << std::endl;
        mut.unlock();
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      mut.lock();
      ball += 1;
      std::cout << name << "! ball is now " << ball << std::endl;
      mut.unlock();

      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });

  return 0;
}

