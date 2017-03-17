#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/experimental/vector.hpp>

void test_fill_constructor()
{
  using namespace agency::experimental;

  {
    // test fill construct empty vector
    vector<int> v(0, 13);

    assert(v.begin() == v.end());
    assert(v.cbegin() == v.cend());
    assert(v.size() == 0);
    assert(v.empty());
  }

  {
    // test fill construct vector
    
    size_t num_elements = 10;

    vector<int> v(num_elements, 13);

    ptrdiff_t expected_difference = num_elements;

    assert(v.end() - v.begin() == expected_difference);
    assert(v.cend() - v.cbegin() == expected_difference);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == 10);
  }
}

int main()
{
  test_fill_constructor();

  std::cout << "OK" << std::endl;

  return 0;
}

