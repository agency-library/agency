#include <vector>
#include <cassert>
#include <numeric>
#include <typeinfo>
#include <functional>

#include <agency/experimental/ranges/zip_with.hpp>

int product(int x, int y, int z)
{
  return x * y * z;
}


void test()
{
  using namespace agency::experimental;

  std::vector<int> v0(4);
  std::vector<int> v1(4);
  std::vector<int> v2(4);

  std::iota(v0.begin(), v0.end(), 0);
  std::iota(v1.begin(), v1.end(), 4);
  std::iota(v2.begin(), v2.end(), 8);

  auto z = zip_with(product, v0,v1,v2);

  {
    // test iterator equality
    assert(z.begin() == z.begin());
  }

  {
    // test iterator inequality
    auto z1 = zip_with(std::plus<int>(), v0,v1);
    auto z2 = zip_with(std::plus<int>(), v1,v2);

    assert(z1.begin() != z2.begin());
  }

  {
    // test sentinel equality
    assert(z.end() == z.end());
  }

  {
    // test iterator/sentinel inequality
    assert(z.begin() != z.end());
  }

  {
    // test iterator/sentinel difference
    assert(z.end() - z.begin() == 4);
  }

  {
    // test iterator increment
    auto iter = z.begin();

    assert(*iter == product(0, 4, 8));

    ++iter;
    assert(*iter == product(1, 5, 9));

    ++iter;
    assert(*iter == product(2, 6, 10));

    ++iter;
    assert(*iter == product(3, 7, 11));
  }

  {
    // test iterator add assign
    auto iter = z.begin();
    assert(*iter == product(0, 4, 8));

    iter += 1;
    assert(*iter == product(1, 5, 9));

    iter += 1;
    assert(*iter == product(2, 6, 10));

    iter += 1;
    assert(*iter == product(3, 7, 11));
  }

  {
    // test iterator/sentinel difference
    auto size = z.end() - z.begin();

    assert(size == 4);
  }

  {
    // test iterator dereference
    assert(*z.begin() == product(0, 4, 8));
  }

  {
    // test iterator index
    auto iter = z.begin();
    assert(iter[0] == product(0, 4, 8));
    assert(iter[1] == product(1, 5, 9));
    assert(iter[2] == product(2, 6, 10));
    assert(iter[3] == product(3, 7, 11));
  }

  {
    // test size
    assert(z.size() == 4);
  }

  {
    // test index
    assert(z[0] == product(0, 4, 8));
    assert(z[1] == product(1, 5, 9));
    assert(z[2] == product(2, 6, 10));
    assert(z[3] == product(3, 7, 11));
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

