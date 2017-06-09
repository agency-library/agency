#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/experimental/vector.hpp>
#include <agency/execution/execution_policy.hpp>

void test_count_constructor()
{
  using namespace agency::experimental;

  {
    // test count construct empty vector

    vector<int> v(0);

    assert(v.end() - v.begin() == 0);
    assert(v.cend() - v.cbegin() == 0);
    assert(v.size() == 0);
    assert(v.empty());
    assert(std::count(v.begin(), v.end(), 0) == 0);
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
}

template<class ExecutionPolicy>
void test_count_constructor(ExecutionPolicy policy)
{
  using namespace agency::experimental;

  {
    // test count construct empty vector
    
    vector<int> v(policy, 0);

    assert(v.end() - v.begin() == 0);
    assert(v.cend() - v.cbegin() == 0);
    assert(v.size() == 0);
    assert(v.empty());
    assert(std::count(v.begin(), v.end(), 0) == 0);
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
}

int main()
{
  test_count_constructor();
  test_count_constructor(agency::seq);
  test_count_constructor(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

