#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>

void test_copy_constructor()
{
  using namespace agency;

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

    ptrdiff_t expected_difference = num_elements;

    assert(other.end() - other.begin() == expected_difference);
    assert(other.cend() - other.cbegin() == expected_difference);
    assert(other.size() == num_elements);
    assert(!other.empty());
    assert(std::count(other.begin(), other.end(), 13) == 10);

    assert(v.end() - v.begin() == expected_difference);
    assert(v.cend() - v.cbegin() == expected_difference);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == 10);
  }
}


template<class ExecutionPolicy>
void test_copy_constructor(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test copy construct empty vector
    vector<int> other;

    vector<int> v(policy, other);

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

    vector<int> v(policy, other);

    ptrdiff_t expected_difference = num_elements;

    assert(other.end() - other.begin() == expected_difference);
    assert(other.cend() - other.cbegin() == expected_difference);
    assert(other.size() == num_elements);
    assert(!other.empty());
    assert(std::count(other.begin(), other.end(), 13) == 10);

    assert(v.end() - v.begin() == expected_difference);
    assert(v.cend() - v.cbegin() == expected_difference);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == 10);
  }
}

int main()
{
  test_copy_constructor();
  test_copy_constructor(agency::seq);
  test_copy_constructor(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

