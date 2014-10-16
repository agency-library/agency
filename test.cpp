#include <iostream>
#include <execution_policy>
#include <mutex>

int main()
{
  using std::seq;
  using std::par;
  using std::con;

  std::bulk_invoke(seq(4), [&](std::sequential_agent &g)
  {
    std::cout << g.index() << std::endl;
  });

  auto f1 = std::bulk_async(seq(4), [&](std::sequential_agent &g)
  {
    std::cout << g.index() << std::endl;
  });

  f1.wait();

  auto f2 = std::bulk_async(seq(2, seq(1)), [&](std::sequential_group<std::sequential_agent> &self)
  {
    std::cout << self.index() << std::endl;
  });

  f2.wait();

  auto f3 = std::bulk_async(seq(3, seq(1, seq(4))), [&](std::sequential_group<std::sequential_group<std::sequential_agent>> &self)
  {
    std::cout << self.index() << std::endl;
  });

  f3.wait();

  return 0;
}

