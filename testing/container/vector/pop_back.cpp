#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/container/vector.hpp>

void test_pop_back()
{
  using namespace agency;

  {
    // test pop_back on single-element vector

    vector<int> v(1,7);

    v.pop_back();

    assert(v.empty());
  }

  {
    // test pop_back on multi-element vector

    size_t initial_size = 10;

    vector<int> v(initial_size, 7);

    auto before_back = v.end() - 2;

    v.pop_back();

    assert(v.size() == initial_size - 1);
    assert(before_back == v.end() - 1);
  }
}

int main()
{
  test_pop_back();

  std::cout << "OK" << std::endl;

  return 0;
}

