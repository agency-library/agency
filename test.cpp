#include <iostream>
#include <execution_policy>
#include <mutex>

int main()
{
  using std::seq;
  using std::par;
  using std::con;

//  auto f = std::bulk_async(seq(4), [&](std::sequential_agent &g)
//  {
//    int i = g.index();
//
//    std::cout << i << std::endl;
//  });
//
//  f.wait();

//  auto f = std::bulk_async(seq(2, seq(1)), [&](std::sequential_group<std::sequential_agent> &self)
//  {
//    int i = self.index();
//    int j = self.child().index();
//
//    std::cout << i << " " << j << std::endl;
//  });
//
//  f.wait();

  auto f = std::bulk_async(seq(3, seq(1, seq(4))), [&](std::sequential_group<std::sequential_group<std::sequential_agent>> &self)
  {
    int i = self.index();
    int j = self.child().index();
    int k = self.child().child().index();

    std::cout << i << " " << j << " " << k << std::endl;
  });

  f.wait();

  return 0;
}

