#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>

#include <agency/experimental/ranges/statically_bounded.hpp>

void test()
{
  using namespace agency::experimental;

  std::vector<int> ints(10);
  std::iota(ints.begin(), ints.end(), 0);

  auto rng = statically_bounded<13>(ints);

  std::vector<int> ref(10);
  std::iota(ref.begin(), ref.end(), 0);

  assert(std::equal(ref.begin(), ref.end(), rng.begin()));

  std::fill(rng.begin(), rng.end(), 13);
  std::fill(ref.begin(), ref.end(), 13);

  assert(std::equal(ref.begin(), ref.end(), rng.begin()));
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

