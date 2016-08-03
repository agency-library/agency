#include <agency/agency.hpp>
#include <mutex>
#include <iostream>
#include <thread>
#include <cassert>

int main()
{
  int ball = 0;
  std::string names[2] = {"ping", "pong"};
  std::mutex mut;

  // create two concurrent agents
  agency::bulk_invoke(agency::con(2), [&](agency::concurrent_agent& self)
  {
    auto name = names[self.index()];

    // play for 20 volleys
    for(int next_state = self.index(); next_state < 20; next_state += 2)
    {
      // wait for the next volley
      while(ball != next_state)
      {
        mut.lock();
        std::cout << name << " waiting for return" << std::endl;
        mut.unlock();
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      // return the ball
      mut.lock();
      ball += 1;
      std::cout << name << "! ball is now " << ball << std::endl;
      mut.unlock();

      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });

  assert(ball == 20);

  std::cout << "OK" << std::endl;

  return 0;
}

