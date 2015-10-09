#include <agency/execution_policy.hpp>
#include <mutex>
#include <iostream>
#include <thread>
#include <cassert>

// this is just like std::mutex, except it has a move constructor
// the move constructor just acts like the default constructor
class movable_mutex
{
  public:
    constexpr movable_mutex() = default;

    movable_mutex(const movable_mutex&) = delete;

    movable_mutex(movable_mutex&&)
      : mut_()
    {}

    void lock()
    {
      mut_.lock();
    }

    void unlock()
    {
      mut_.unlock();
    }

  private:
    std::mutex mut_;
};

bool ping_pong_match(agency::concurrent_agent& self, const std::vector<std::string>& names, int num_volleys, movable_mutex& mut, int& ball)
{
  auto name = names[self.index()];
  
  for(int next_state = self.index();
      (next_state < num_volleys) && (ball >= 0);
      next_state += 2)
  {
    // wait for the next volley
    while(ball >= 0 && ball != next_state)
    {
      mut.lock();
      std::cout << name << " waiting for return" << std::endl;
      mut.unlock();
      
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  
    mut.lock();
    if(ball == -1)
    {
      std::cout << name << " wins!" << std::endl;
    }
    else
    {
      // try to return the ball
      if(std::rand() % 10 == 0)
      {
        // whiff -- declare outself the loser
        ball = -self.index() - 1;
        std::cout << "whiff... " << name << " loses!" << std::endl;
      }
      else
      {
        // successful return
        ball += 1;
        std::cout << name << "! ball is now " << ball << std::endl;
      }
    }
    mut.unlock();
  
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  // decode who lost the match
  int loser = (ball > 0) ? 1 : -(ball + 1);

  // return the winner
  return loser ^ 1;
}

int main()
{
  size_t num_volleys = 20;
  std::vector<std::string> names = {"ping", "pong"};

  agency::bulk_invoke(agency::con(2), ping_pong_match, names, num_volleys, agency::share<0,movable_mutex>(), agency::share<0,int>());

  // XXX figure out how to communicate the winner, if possible
//  assert(ball == 20 || ball == -1);
//
//  if(ball == 20)
//  {
//    std::cout << "It's a tie... ping wins!" << std::endl;
//  }

  std::cout << "OK" << std::endl;

  return 0;
}

