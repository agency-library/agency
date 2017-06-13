#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <list>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>

template<class Container>
void test_reallocating_range_insert()
{
  using namespace agency;

  using value_type = typename Container::value_type;

  {
    // test range insert into empty vector

    vector<value_type> v;

    size_t num_elements_to_insert = 5;
    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_elements_to_insert);
    assert(std::equal(v.begin(), v.end(), items.begin()));
  }

  {
    // test range insert at the beginning of vector

    size_t num_initial_elements = 10;
    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test range insert at the end of vector
    
    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.end(), items.begin(), items.end());

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test range insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, items.begin(), items.end());
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(insertion_begin, insertion_end, items.begin()));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


template<class Container, class ExecutionPolicy>
void test_reallocating_range_insert(ExecutionPolicy policy)
{
  using namespace agency;

  using value_type = typename Container::value_type;

  {
    // test range insert into empty vector

    vector<value_type> v;

    size_t num_elements_to_insert = 5;
    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(policy, v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_elements_to_insert);
    assert(std::equal(v.begin(), v.end(), items.begin()));
  }

  {
    // test range insert at the beginning of vector

    size_t num_initial_elements = 10;
    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(policy, v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test range insert at the end of vector
    
    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(policy, v.end(), items.begin(), items.end());

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test range insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(policy, middle, items.begin(), items.end());
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(insertion_begin, insertion_end, items.begin()));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


template<class Container>
void test_nonreallocating_range_insert()
{
  using namespace agency;

  using value_type = typename Container::value_type;

  {
    // test range insert into beginning of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }
  
  {
    // test range insert at the end of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.end(), items.begin(), items.end());

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test range insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, items.begin(), items.end());
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(insertion_begin, insertion_end, items.begin()));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


template<class Container, class ExecutionPolicy>
void test_nonreallocating_range_insert(ExecutionPolicy policy)
{
  using namespace agency;

  using value_type = typename Container::value_type;

  {
    // test range insert into beginning of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(policy, v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }
  
  {
    // test range insert at the end of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(policy, v.end(), items.begin(), items.end());

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test range insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<value_type> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    Container items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(policy, middle, items.begin(), items.end());
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(insertion_begin, insertion_end, items.begin()));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


int main()
{
  {
    // test insertion from std::vector

    test_reallocating_range_insert<std::vector<int>>();
    test_reallocating_range_insert<std::vector<int>>(agency::par);

    test_nonreallocating_range_insert<std::vector<int>>();
    test_nonreallocating_range_insert<std::vector<int>>(agency::par);
  }

  {
    // test insertion from std::list

    // list has iterators we can't parallelize, and the implementation of .insert() with
    // no execution policy parameter uses agency::seq internally, so there's no reason
    // to test std::list with policy parameter
    test_reallocating_range_insert<std::list<int>>();
    test_nonreallocating_range_insert<std::list<int>>();
  }

  std::cout << "OK" << std::endl;

  return 0;
}

