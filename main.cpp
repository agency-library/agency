#include <iostream>
#include <execution_policy>

std::mutex mut;

int main()
{
  using std::seq;
  using std::par;
  using std::con;

  auto seq_f = bulk_async(seq(5), [](std::sequential_agent<> &g)
  {
    std::cout << "Hello world from sequential_agent " << g.index() << std::endl;
  });

  auto par_f = bulk_async(par(5), [](std::parallel_agent<> &g)
  {
    mut.lock();
    std::cout << "Hello world from parallel_agent " << g.index() << g.index() << std::endl;
    mut.unlock();
  });

  auto con_f = bulk_async(con(5), [](std::concurrent_agent<> &g)
  {
    mut.lock();
    std::cout << "Hello world from concurrent_agent " << g.index() << " arriving at barrier." << std::endl;
    mut.unlock();

    g.wait();

    mut.lock();
    std::cout << "Hello world from concurrent_agent " << g.index() << " departing from barrier." << std::endl;
    mut.unlock();
  });

  seq_f.wait();
  par_f.wait();
  con_f.wait();

  auto singly_nested_f = bulk_async(con(2, seq(3)), [](std::concurrent_agent<std::sequential_agent<>> &g)
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

  auto doubly_nested_f = bulk_async(seq(2, par(2, seq(3))), [](std::sequential_agent<std::parallel_agent<std::sequential_agent<>>> &g)
  {
    mut.lock();
    std::cout << "Hello world from sequential_agent " << g.child().child().index() << " of parallel_agent " << g.child().index() << " of sequential_agent " << g.index() << std::endl;
    mut.unlock();
  });

  doubly_nested_f.wait();

  return 0;
}

