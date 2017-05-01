#include <vector>
#include <cassert>
#include <numeric>
#include <typeinfo>
#include <functional>

#include <agency/experimental/ranges/transformed.hpp>

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

  auto t = transformed(product, v0,v1,v2);

  {
    // test iterator equality
    assert(t.begin() == t.begin());
  }

  {
    // test iterator inequality
    auto t1 = transformed(std::plus<int>(), v0,v1);
    auto t2 = transformed(std::plus<int>(), v1,v2);

    assert(t1.begin() != t2.begin());
  }

  {
    // test sentinel equality
    assert(t.end() == t.end());
  }

  {
    // test iterator/sentinel inequality
    assert(t.begin() != t.end());
  }

  {
    // test iterator/sentinel difference
    assert(t.end() - t.begin() == 4);
  }

  {
    // test iterator increment
    auto iter = t.begin();

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
    auto iter = t.begin();
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
    auto size = t.end() - t.begin();

    assert(size == 4);
  }

  {
    // test iterator dereference
    assert(*t.begin() == product(0, 4, 8));
  }

  {
    // test iterator index
    auto iter = t.begin();
    assert(iter[0] == product(0, 4, 8));
    assert(iter[1] == product(1, 5, 9));
    assert(iter[2] == product(2, 6, 10));
    assert(iter[3] == product(3, 7, 11));
  }

  {
    // test size
    assert(t.size() == 4);
  }

  {
    // test index
    assert(t[0] == product(0, 4, 8));
    assert(t[1] == product(1, 5, 9));
    assert(t[2] == product(2, 6, 10));
    assert(t[3] == product(3, 7, 11));
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

