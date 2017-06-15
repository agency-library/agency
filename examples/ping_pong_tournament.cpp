#include <agency/agency.hpp>
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


// this function implements a single concurrent ping pong match
size_t ping_pong_match(agency::concurrent_agent& self, const std::vector<std::string>& names, int num_volleys, movable_mutex& mut, int& ball)
{
  // agent 0 initializes the ball
  if(self.index() == 0)
  {
    ball = 0;
  }

  self.wait();

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


// this function implements a two concurrent semifinals ping pong matches
// followed by a single final ping pong match
void ping_pong_tournament(agency::concurrent_group<agency::concurrent_agent>& self,
                          const std::vector<std::vector<std::string>>& semifinalists,
                          int num_volleys,
                          movable_mutex& mut,
                          std::vector<std::string>& finalists,
                          int& ball)
{
  if(self.inner().index() == 0)
  {
    if(self.outer().index() == 0)
    {
      std::cout << "Starting semifinal matches..." << std::endl;
    }

    self.outer().wait();
  }

  self.inner().wait();

  // play the semifinals matches
  auto semifinal_winner_idx = ping_pong_match(self.inner(), semifinalists[self.outer().index()], num_volleys, mut, ball);

  if(self.inner().index() == 0)
  {
    // the first agent of each group reports who won the semifinal match
    auto semifinal_winner = semifinalists[self.outer().index()][semifinal_winner_idx];
    finalists[self.outer().index()] = semifinal_winner;

    // have the first player of each group wait for the other group
    self.outer().wait();
  }

  // have each inner group wait for each other
  self.inner().wait();

  // group 0 plays the final match while group 1 sits it out
  if(self.outer().index() == 0)
  {
    // agent 0 initializes the contestant names for the final match
    if(self.inner().index() == 0)
    {
      std::cout << std::endl << finalists[0] << " and " << finalists[1] << " starting the final match..." << std::endl;
    }

    // wait until agent 0 initializes the contestant names before starting the final match
    self.inner().wait();

    // play the final match
    auto final_winner_idx = ping_pong_match(self.inner(), finalists, num_volleys, mut, ball);

    // have agent 0 of group 0 report the winner
    if(self.inner().index() == 0)
    {
      std::cout << std::endl << finalists[final_winner_idx] << " is the tournament champion!" << std::endl;
    }
  }
}


int main()
{
  using namespace agency;

  size_t num_volleys = 20;
  std::vector<std::vector<std::string>> semifinalists = {{"ping", "pong"}, {"foo", "bar"}};

  bulk_invoke(con(2, con(2)), ping_pong_tournament, semifinalists, num_volleys, share<movable_mutex>(), share<std::vector<std::string>>(2), share_at_scope<1,int>());

  return 0;
}

