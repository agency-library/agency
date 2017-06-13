#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>

void test_reallocating_fill_insert()
{
  using namespace agency;

  {
    // test fill insert into empty vector

    vector<int> v;

    size_t num_elements_to_insert = 5;

    auto result = v.insert(v.begin(), num_elements_to_insert, 7);

    assert(result == v.begin());
    assert(v.size() == num_elements_to_insert);
    assert(std::count(v.begin(), v.end(), 7) == static_cast<int>(num_elements_to_insert));
  }

  {
    // test fill insert at the beginning of vector
    
    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    auto result = v.insert(v.begin(), num_elements_to_insert, 7);

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test fill insert at the end of vector
    
    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    auto result = v.insert(v.end(), num_elements_to_insert, 7);

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test fill insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, num_elements_to_insert, 7);
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(insertion_begin, insertion_end, 7) == static_cast<int>(num_elements_to_insert));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


template<class ExecutionPolicy>
void test_reallocating_fill_insert(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test fill insert into empty vector

    vector<int> v;

    size_t num_elements_to_insert = 5;

    auto result = v.insert(policy, v.begin(), num_elements_to_insert, 7);

    assert(result == v.begin());
    assert(v.size() == num_elements_to_insert);
    assert(std::count(v.begin(), v.end(), 7) == static_cast<int>(num_elements_to_insert));
  }

  {
    // test fill insert at the beginning of vector
    
    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    auto result = v.insert(policy, v.begin(), num_elements_to_insert, 7);

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test fill insert at the end of vector
    
    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    auto result = v.insert(policy, v.end(), num_elements_to_insert, 7);

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test fill insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(policy, middle, num_elements_to_insert, 7);
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(insertion_begin, insertion_end, 7) == static_cast<int>(num_elements_to_insert));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


void test_nonreallocating_fill_insert()
{
  using namespace agency;

  {
    // test fill insert into beginning of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    auto result = v.insert(v.begin(), num_elements_to_insert, 7);

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }
  
  {
    // test fill insert at the end of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    auto result = v.insert(v.end(), num_elements_to_insert, 7);

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test fill insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, num_elements_to_insert, 7);
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(insertion_begin, insertion_end, 7) == static_cast<int>(num_elements_to_insert));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


template<class ExecutionPolicy>
void test_nonreallocating_fill_insert(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test fill insert into beginning of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    auto result = v.insert(policy, v.begin(), num_elements_to_insert, 7);

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == static_cast<int>(num_initial_elements));
  }
  
  {
    // test fill insert at the end of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    auto result = v.insert(policy, v.end(), num_elements_to_insert, 7);

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == static_cast<int>(num_elements_to_insert));
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
  }

  {
    // test fill insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(policy, middle, num_elements_to_insert, 7);
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(insertion_begin, insertion_end, 7) == static_cast<int>(num_elements_to_insert));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}


int main()
{
  test_reallocating_fill_insert();
  test_reallocating_fill_insert(agency::par);

  test_nonreallocating_fill_insert();
  test_nonreallocating_fill_insert(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

