#include <execution_policy>
#include <iostream>

std::mutex mut;

int main()
{
  using std::seq;
  using std::par;
  using std::con;

  auto seq_f = async(seq(5), [](std::sequential_group &g)
  {
    std::cout << "Hello world from agent " << g.child().index() << " in sequential_group " << g.index() << std::endl;
  });

  auto par_f = async(par(5), [](std::parallel_group &g)
  {
    mut.lock();
    std::cout << "Hello world from agent " << g.child().index() << " in parallel_group " << g.index() << std::endl;
    mut.unlock();
  });

  auto con_f = async(con(5), [](std::concurrent_group &g)
  {
    mut.lock();
    std::cout << "Hello world from agent " << g.child().index() << " in concurrent_group " << g.index() << " arriving at barrier." << std::endl;
    mut.unlock();

    g.wait();

    mut.lock();
    std::cout << "Hello world from agent " << g.child().index() << " in concurrent_group " << g.index() << " departing from barrier." << std::endl;
    mut.unlock();
  });

  seq_f.wait();
  par_f.wait();
  con_f.wait();

  auto singly_nested_f = async(con(2, seq(3)), [](std::basic_concurrent_group<std::sequential_group> &g)
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

  auto doubly_nested_f = async(seq(2, par(2, seq(3))), [](std::basic_sequential_group<std::basic_parallel_group<std::sequential_group>> &g)
  {
    mut.lock();
    std::cout << "Hello world from agent " << g.child().child().child().index() << " in sequential_group " << g.child().child().index() << " of parallel_group " << g.child().index() << " of sequential_group " << g.index() << std::endl;
    mut.unlock();

    // query the runtime for the group
    using std::this_group;

    mut.lock();
    std::cout << "Hello world from agent " << this_group().child().child().child().index() << " in sequential_group " << this_group().child().child().index() << " of parallel_group " << this_group().child().index() << " of sequential_group " << this_group().index() << std::endl;
    mut.unlock();
  });

  doubly_nested_f.wait();

  return 0;
}

