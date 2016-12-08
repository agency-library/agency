#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>

#include <agency/experimental/ranges/interval.hpp>

void test()
{
  using namespace agency::experimental;

  size_t n = 100;

  auto rng = interval(10, 20);

  std::vector<int> ref(n);
  std::iota(ref.begin(), ref.end(), 10);

  assert(std::equal(ref.begin(), ref.end(), rng.begin()));
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

