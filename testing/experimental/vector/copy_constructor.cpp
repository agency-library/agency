#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/experimental/vector.hpp>

void test_copy_constructor()
{
  using namespace agency::experimental;

  {
    // test copy construct empty vector
    vector<int> other;

    vector<int> v = other;

    assert(v.begin() == v.end());
    assert(v.cbegin() == v.cend());
    assert(v.size() == 0);
    assert(v.empty());
  }

  {
    // test copy construct non-empty vector
    
    size_t num_elements = 10;

    vector<int> other(num_elements, 13);

    vector<int> v = other;

    assert(v.end() - v.begin() == num_elements);
    assert(v.cend() - v.cbegin() == num_elements);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == 10);
  }
}

int main()
{
  test_copy_constructor();

  std::cout << "OK" << std::endl;

  return 0;
}

