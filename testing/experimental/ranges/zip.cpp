#include <vector>
#include <cassert>
#include <numeric>
#include <typeinfo>

#include <agency/experimental/ranges/zip.hpp>

template<class Tuple>
void assign_second_to_first(const Tuple& t)
{
  agency::get<0>(t) = agency::get<1>(t);
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

  auto z = zip(v0,v1,v2);

  {
    // test iterator equality
    assert(z.begin() == z.begin());
  }

  {
    // test iterator inequality
    auto z1 = zip(v0,v1);
    auto z2 = zip(v1,v2);

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

    assert(*iter == agency::make_tuple(0, 4, 8));

    ++iter;
    assert(*iter == agency::make_tuple(1, 5, 9));

    ++iter;
    assert(*iter == agency::make_tuple(2, 6, 10));

    ++iter;
    assert(*iter == agency::make_tuple(3, 7, 11));
  }

  {
    // test iterator add assign
    auto iter = z.begin();
    assert(*iter == agency::make_tuple(0, 4, 8));

    iter += 1;
    assert(*iter == agency::make_tuple(1, 5, 9));

    iter += 1;
    assert(*iter == agency::make_tuple(2, 6, 10));

    iter += 1;
    assert(*iter == agency::make_tuple(3, 7, 11));
  }

  {
    // test iterator/sentinel difference
    auto size = z.end() - z.begin();

    assert(size == 4);
  }

  {
    // test iterator dereference
    assert(*z.begin() == agency::make_tuple(0, 4, 8));
  }

  {
    // test assign through iterator dereference
    std::vector<int> v1(4);
    std::vector<int> v2(4);

    std::iota(v1.begin(), v1.end(), 0);
    std::iota(v2.begin(), v2.end(), 4);

    std::vector<int> v3(4,0);
    std::vector<int> v4(4,0);

    auto z1 = zip(v1,v2);
    auto z2 = zip(v3,v4);

    auto i1 = z1.begin();
    auto i2 = z2.begin();

    *i2 = *i1;
    ++i2;
    ++i1;
    
    *i2 = *i1;
    ++i2;
    ++i1;

    *i2 = *i1;
    ++i2;
    ++i1;

    *i2 = *i1;

    assert(v3[0] == 0);
    assert(v3[1] == 1);
    assert(v3[2] == 2);
    assert(v3[3] == 3);

    assert(v4[0] == 4);
    assert(v4[1] == 5);
    assert(v4[2] == 6);
    assert(v4[3] == 7);
  }

  {
    // test iterator index
    auto iter = z.begin();
    assert(iter[0] == agency::make_tuple(0, 4, 8));
    assert(iter[1] == agency::make_tuple(1, 5, 9));
    assert(iter[2] == agency::make_tuple(2, 6, 10));
    assert(iter[3] == agency::make_tuple(3, 7, 11));
  }

  {
    // test assign through iterator index
    std::vector<int> v1(4);
    std::vector<int> v2(4);

    std::iota(v1.begin(), v1.end(), 0);
    std::iota(v2.begin(), v2.end(), 4);

    std::vector<int> v3(4,0);
    std::vector<int> v4(4,0);

    auto z1 = zip(v1,v2);
    auto z2 = zip(v3,v4);

    for(int i = 0; i < z1.size(); ++i)
    {
      z2[i] = z1[i];
    }

    assert(v3[0] == 0);
    assert(v3[1] == 1);
    assert(v3[2] == 2);
    assert(v3[3] == 3);

    assert(v4[0] == 4);
    assert(v4[1] == 5);
    assert(v4[2] == 6);
    assert(v4[3] == 7);
  }

  {
    // test size
    assert(z.size() == 4);
  }

  {
    // test index
    assert(z[0] == agency::make_tuple(0, 4, 8));
    assert(z[1] == agency::make_tuple(1, 5, 9));
    assert(z[2] == agency::make_tuple(2, 6, 10));
    assert(z[3] == agency::make_tuple(3, 7, 11));
  }

  {
    // test assign through index
    std::vector<int> v1(4);
    std::vector<int> v2(4);

    std::iota(v1.begin(), v1.end(), 0);
    std::iota(v2.begin(), v2.end(), 4);

    std::vector<int> v3(4,0);
    std::vector<int> v4(4,0);

    auto z1 = zip(v1,v2);
    auto z2 = zip(v3,v4);

    for(int i = 0; i < z.size(); ++i)
    {
      z2[i] = z1[i];
    }

    assert(v3[0] == 0);
    assert(v3[1] == 1);
    assert(v3[2] == 2);
    assert(v3[3] == 3);

    assert(v4[0] == 4);
    assert(v4[1] == 5);
    assert(v4[2] == 6);
    assert(v4[3] == 7);
  }

  {
    // test assign second to first
    std::vector<int> v1(4);
    std::vector<int> v2(4);

    std::iota(v1.begin(), v1.end(), 0);
    std::iota(v2.begin(), v2.end(), 4);

    auto z1 = zip(v1,v2);

    for(int i = 0; i < z.size(); ++i)
    {
      assign_second_to_first(z1[i]);
    }

    assert(v1[0] == 4);
    assert(v1[1] == 5);
    assert(v1[2] == 6);
    assert(v1[3] == 7);

    assert(v2[0] == 4);
    assert(v2[1] == 5);
    assert(v2[2] == 6);
    assert(v2[3] == 7);
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

