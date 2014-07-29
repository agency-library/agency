#include <iostream>
#include <cassert>
#include <algorithm>
#include <execution_policy>

int main()
{
  size_t n = 1 << 16;
  std::vector<int> x(n);

  auto f = std::bulk_async(std::par(n), [&](std::parallel_group<> &g)
  {
    int i = g.child().index();
    x[i] = 13;
  });

  try
  {
    f.get();
  }
  catch(std::exception_list &e)
  {
    std::cerr << "caught std::exception_list: " << e.what() << std::endl;
    std::terminate();
  }

  for(int i = 0; i < n; ++i)
  {
    if(x[i] != 13)
    {
      std::cout << "x[" << i << "]: " << x[i] << std::endl;
    }
  }

  assert(std::all_of(x.begin(), x.end(), [=](int x){ return 13 == x; }));

  std::cout << "OK" << std::endl;

  return 0;
}

