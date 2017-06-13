#include <iostream>
#include <cassert>
#include <algorithm>
#include <utility>
#include <agency/container/vector.hpp>

void test_move_constructor()
{
  using namespace agency;

  {
    // test move construct empty vector
    vector<int> other;

    vector<int> v = std::move(other);

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
    // test move construct non-empty vector
    
    size_t num_elements = 10;

    vector<int> other(num_elements, 13);

    vector<int> v = std::move(other);

    assert(other.begin() == other.end());
    assert(other.cbegin() == other.cend());
    assert(other.size() == 0);
    assert(other.empty());

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
  test_move_constructor();

  std::cout << "OK" << std::endl;

  return 0;
}

