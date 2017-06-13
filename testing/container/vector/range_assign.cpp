#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <list>
#include <agency/container/vector.hpp>

template<class Container>
void test_reallocating_range_assign()
{
  using namespace agency;

  {
    // test range assign into empty vector

    vector<int> v;

    size_t num_elements_to_assign = 5;
    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into small vector

    vector<int> v(3);

    size_t num_elements_to_assign = 5;
    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }
}

template<class Container, class ExecutionPolicy>
void test_reallocating_range_assign(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test range assign into empty vector

    vector<int> v;

    size_t num_elements_to_assign = 5;
    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(policy, assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into small vector

    vector<int> v(3);

    size_t num_elements_to_assign = 5;
    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(policy, assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }
}

template<class Container>
void test_nonreallocating_range_assign()
{
  using namespace agency;

  using value_type = typename Container::value_type;

  {
    // test range assign into empty vector with capacity

    vector<value_type> v;
    
    size_t num_elements_to_assign = 5;
    v.reserve(5);

    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into small vector with capacity

    vector<value_type> v(3);
    
    size_t num_elements_to_assign = 5;
    v.reserve(5);

    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into large vector
    size_t num_elements_to_assign = 5;
    
    vector<value_type> v(2 * num_elements_to_assign);

    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }
}

template<class Container, class ExecutionPolicy>
void test_nonreallocating_range_assign(ExecutionPolicy policy)
{
  using namespace agency;

  using value_type = typename Container::value_type;

  {
    // test range assign into empty vector with capacity

    vector<value_type> v;
    
    size_t num_elements_to_assign = 5;
    v.reserve(5);

    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(policy, assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into small vector with capacity

    vector<value_type> v(3);
    
    size_t num_elements_to_assign = 5;
    v.reserve(5);

    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(policy, assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }

  {
    // test range assign into large vector
    size_t num_elements_to_assign = 5;
    
    vector<value_type> v(2 * num_elements_to_assign);

    Container assign_me(num_elements_to_assign);
    std::iota(assign_me.begin(), assign_me.end(), 0);

    v.assign(policy, assign_me.begin(), assign_me.end());

    assert(v.size() == num_elements_to_assign);
    assert(std::equal(v.begin(), v.end(), assign_me.begin()));
  }
}

int main()
{
  {
    // test assign from std::vector

    test_reallocating_range_assign<std::vector<int>>();
    test_reallocating_range_assign<std::vector<int>>(agency::par);

    test_nonreallocating_range_assign<std::vector<int>>();
    test_nonreallocating_range_assign<std::vector<int>>(agency::par);
  }

  {
    // test assign from std::list
    
    test_reallocating_range_assign<std::list<int>>();
    test_nonreallocating_range_assign<std::list<int>>();
  }

  std::cout << "OK" << std::endl;

  return 0;
}

