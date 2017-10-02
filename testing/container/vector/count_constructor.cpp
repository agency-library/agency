#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>

struct move_only
{
  int value;
  int& reference;

  move_only()
    : value{},
      reference{value}
  {}

  bool operator==(int other) const
  {
    return value == other;
  }
};


void test_count_constructor()
{
  using namespace agency;

  {
    // test count construct empty vector
    vector<int> v(0);

    assert(v.begin() == v.end());
    assert(v.cbegin() == v.cend());
    assert(v.size() == 0);
    assert(v.empty());
  }

  {
    // test count construct vector
    
    size_t num_elements = 10;

    vector<int> v(num_elements);

    ptrdiff_t expected_difference = num_elements;

    assert(v.end() - v.begin() == expected_difference);
    assert(v.cend() - v.cbegin() == expected_difference);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 0) == 10);
  }

  {
    // test count construct vector of move_only elements
    size_t num_elements = 10;

    vector<move_only> v(num_elements);

    ptrdiff_t expected_difference = num_elements;

    assert(v.end() - v.begin() == expected_difference);
    assert(v.cend() - v.cbegin() == expected_difference);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 0) == 10);
  }
}

template<class ExecutionPolicy>
void test_count_constructor(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test count construct empty vector
    vector<int> v(policy, 0);

    assert(v.begin() == v.end());
    assert(v.cbegin() == v.cend());
    assert(v.size() == 0);
    assert(v.empty());
  }

  {
    // test count construct vector
    
    size_t num_elements = 10;

    vector<int> v(policy, num_elements);

    ptrdiff_t expected_difference = num_elements;

    assert(v.end() - v.begin() == expected_difference);
    assert(v.cend() - v.cbegin() == expected_difference);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 0) == 10);
  }

  {
    // test count construct vector of move_only elements
    size_t num_elements = 10;

    vector<move_only> v(num_elements);

    ptrdiff_t expected_difference = num_elements;

    assert(v.end() - v.begin() == expected_difference);
    assert(v.cend() - v.cbegin() == expected_difference);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 0) == 10);
  }
}

int main()
{
  test_count_constructor();
  test_count_constructor(agency::seq);
  test_count_constructor(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

