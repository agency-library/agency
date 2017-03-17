#include <agency/agency.hpp>
#include <iostream>
#include <cassert>
#include <algorithm>

int main()
{
  using namespace agency;

  size_t n = 1 << 16;
  std::vector<int> x(n);

  auto f = bulk_async(par(n), [&](parallel_agent &self)
  {
    int i = self.index();
    x[i] = 13;
  });

  try
  {
    f.get();
  }
  catch(exception_list &e)
  {
    std::cerr << "caught exception_list: " << e.what() << std::endl;
    std::terminate();
  }

  for(size_t i = 0; i < n; ++i)
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

