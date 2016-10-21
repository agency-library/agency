#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/experimental/vector.hpp>

void test_single_insert_copy()
{
  using namespace agency::experimental;

  {
    // test single insert copy into empty vector

    vector<int> v;

    int insert_me = 7;
    auto result = v.insert(v.begin(), insert_me);

    assert(result == v.begin());
    assert(*result == 7);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), 7) == 1);
  }

  {
    // test single insert at the beginning of vector
    
    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    int insert_me = 7;
    auto result = v.insert(v.begin(), insert_me);

    assert(result == v.begin());
    assert(*result == 7);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.begin(), v.end(), 7) == 1);
  }

  {
    // test single insert at the end of vector
    
    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    int insert_me = 7;
    auto result = v.insert(v.end(), insert_me);

    assert(result == v.end() - 1);
    assert(*result == 7);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(result, result + 1, 7) == 1);
    assert(std::count(v.begin(), result, 13) == num_initial_elements);
  }

  {
    // test single insert in the middle of vector

    size_t num_initial_elements = 10;

    vector<int> v(num_initial_elements, 13);

    int insert_me = 7;

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, insert_me);
    auto insertion_end = insertion_begin + 1;

    assert(*insertion_begin == 7);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(insertion_begin, insertion_end, 7) == 1);

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, 13) == num_initial_elements_before);
    assert(std::count(insertion_end, v.end(), 13) == num_initial_elements_after);
  }
}

int main()
{
  test_single_insert_copy();

  std::cout << "OK" << std::endl;

  return 0;
}

