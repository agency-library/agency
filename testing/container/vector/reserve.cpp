#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>

void test_reserve()
{
  using namespace agency;

  {
    // test reserve on initially empty vector

    vector<int> v;
    size_t reservation = 100;

    v.reserve(reservation);

    assert(v.capacity() >= reservation);
    assert(v.size() == 0);
    assert(v.empty());
  }

  {
    // test reserve on initially non-empty vector

    size_t num_initial_elements = 10;
    vector<int> v(num_initial_elements, 13);

    assert(std::count(v.begin(), v.end(), 13) == static_cast<int>(num_initial_elements));

    size_t reservation = 100;

    v.reserve(reservation);

    assert(v.capacity() >= reservation);
    assert(v.size() == num_initial_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == static_cast<int>(num_initial_elements));
  }
}


template<class ExecutionPolicy>
void test_reserve(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test reserve on initially empty vector

    vector<int> v;
    size_t reservation = 100;

    v.reserve(policy, reservation);

    assert(v.capacity() >= reservation);
    assert(v.size() == 0);
    assert(v.empty());
  }

  {
    // test reserve on initially non-empty vector

    size_t num_initial_elements = 10;
    vector<int> v(num_initial_elements, 13);

    assert(std::count(v.begin(), v.end(), 13) == static_cast<int>(num_initial_elements));

    size_t reservation = 100;

    v.reserve(policy, reservation);

    assert(v.capacity() >= reservation);
    assert(v.size() == num_initial_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == static_cast<int>(num_initial_elements));
  }
}

int main()
{
  test_reserve();
  test_reserve(agency::seq);
  test_reserve(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

