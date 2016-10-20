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

    assert(other.begin() == other.end());
    assert(other.cbegin() == other.cend());
    assert(other.size() == 0);
    assert(other.empty());

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

    assert(other.end() - other.begin() == num_elements);
    assert(other.cend() - other.cbegin() == num_elements);
    assert(other.size() == num_elements);
    assert(!other.empty());
    assert(std::count(other.begin(), other.end(), 13) == 10);

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

