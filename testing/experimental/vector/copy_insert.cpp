#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <agency/experimental/vector.hpp>

void test_reallocating_copy_insert()
{
  using namespace agency::experimental;

  {
    // test copy insert into empty vector

    vector<int> v;

    size_t num_elements_to_insert = 5;
    std::vector<int> items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_elements_to_insert);
    assert(std::equal(v.begin(), v.end(), items.begin()));
  }

  {
    // test copy insert at the beginning of vector

    size_t num_initial_elements = 10;
    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    std::vector<int> items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == num_initial_elements);
  }

  {
    // test copy insert at the end of vector
    
    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;

    std::vector<int> items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.end(), items.begin(), items.end());

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(v.begin(), result, 13) == num_initial_elements);
  }

  {
    // test copy insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    std::vector<int> items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, items.begin(), items.end());
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(insertion_begin, insertion_end, items.begin()));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == num_initial_elements_before);
    assert(std::count(insertion_end, v.end(), 13) == num_initial_elements_after);
  }
}


void test_nonreallocating_copy_insert()
{
  using namespace agency::experimental;

  {
    // test copy insert into beginning of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    std::vector<int> items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.begin(), items.begin(), items.end());

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == num_initial_elements);
  }
  
  {
    // test copy insert at the end of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    std::vector<int> items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto result = v.insert(v.end(), items.begin(), items.end());

    assert(result == v.end() - num_elements_to_insert);
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(result, result + num_elements_to_insert, items.begin()));
    assert(std::count(v.begin(), result, 13) == num_initial_elements);
  }

  {
    // test copy insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    v.reserve(num_initial_elements + num_elements_to_insert);

    std::vector<int> items(num_elements_to_insert);
    std::iota(items.begin(), items.end(), 0);

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, items.begin(), items.end());
    auto insertion_end = insertion_begin + num_elements_to_insert;

    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::equal(insertion_begin, insertion_end, items.begin()));

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == num_initial_elements_before);
    assert(std::count(insertion_end, v.end(), 13) == num_initial_elements_after);
  }
}


int main()
{
  //test_reallocating_copy_insert();
  test_nonreallocating_copy_insert();

  std::cout << "OK" << std::endl;

  return 0;
}

