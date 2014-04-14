#include <iostream>
#include <execution_policy>
#include <mutex>

int main()
{
  using std::seq;
  using std::par;
  using std::con;

//  auto f = std::async(seq(2, seq(1)), [&](std::basic_sequential_group<std::sequential_group> &g)
//  {
//    int i = g.child().index();
//    int j = g.child().child().index();
//
//    std::cout << i << " " << j << std::endl;
//  });
//
//  f.wait();

//  auto f = std::async(seq(1, seq(2)), [&](std::basic_sequential_group<std::sequential_group> &g)
//  {
//    int i = g.child().index();
//    int j = g.child().child().index();
//
//    std::cout << i << " " << j << std::endl;
//  });
//
//  f.wait();

//  auto f = std::async(seq(2), [&](std::sequential_group &g)
//  {
//    int i = g.child().index();
//
//    std::cout << i << std::endl;
//  });
//
//  f.wait();

  auto f = std::async(seq(3, seq(1, seq(4))), [&](std::basic_sequential_group<std::basic_sequential_group<std::sequential_group>> &g)
  {
    int i = g.child().index();
    int j = g.child().child().index();
    int k = g.child().child().child().index();

    std::cout << i << " " << j << " " << k << std::endl;
  });

  f.wait();

  return 0;
}

