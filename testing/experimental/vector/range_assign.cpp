#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <agency/experimental/vector.hpp>

void test_reallocating_range_assign()
{
  using namespace agency::experimental;

  {
    // test range assign into empty vector

    vector<int> v;

    size_t num_elements_to_assign = 5;
    std::vector<int> assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into small vector

    vector<int> v(3);

    size_t num_elements_to_assign = 5;
    std::vector<int> assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }
}

void test_nonreallocating_range_assign()
{
  using namespace agency::experimental;

  {
    // test range assign into empty vector with capacity

    vector<int> v;
    
    size_t num_elements_to_assign = 5;
    v.reserve(5);

    std::vector<int> assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into small vector with capacity

    vector<int> v(3);
    
    size_t num_elements_to_assign = 5;
    v.reserve(5);

    std::vector<int> assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into large vector
    size_t num_elements_to_assign = 5;
    
    vector<int> v(2 * num_elements_to_assign);

    std::vector<int> assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }
}

int main()
{
  test_reallocating_range_assign();
  test_nonreallocating_range_assign();

  std::cout << "OK" << std::endl;

  return 0;
}

