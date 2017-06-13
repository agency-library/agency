#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <agency/container/vector.hpp>

void test_single_insert_copy()
{
  using namespace agency;

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
    assert(std::count(v.begin(), result, 13) == static_cast<int>(num_initial_elements));
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

    assert(std::count(v.begin(), insertion_begin, 13) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), 13) == static_cast<int>(num_initial_elements_after));
  }
}

void test_single_insert_move()
{
  using namespace agency;

  {
    // test single insert move into empty vector
    std::vector<int> inserted_value(1,7);

    vector<std::vector<int>> v;

    std::vector<int> insert_me = inserted_value;
    auto result = v.insert(v.begin(), std::move(insert_me));

    assert(insert_me.empty());
    assert(result == v.begin());
    assert(*result == inserted_value);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), inserted_value) == 1);
  }

  {
    // test single insert move at the beginning of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> inserted_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> insert_me = inserted_value;
    auto result = v.insert(v.begin(), std::move(insert_me));

    assert(insert_me.empty());
    assert(result == v.begin());
    assert(*result == inserted_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.begin(), v.end(), inserted_value) == 1);
  }

  {
    // test single insert move at the end of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> inserted_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> insert_me = inserted_value;
    auto result = v.insert(v.end(), std::move(insert_me));

    assert(insert_me.empty());
    assert(result == v.end() - 1);
    assert(*result == inserted_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(result, result + 1, inserted_value) == 1);
    assert(std::count(v.begin(), result, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test single insert in the middle of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> inserted_value(1,7);

    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> insert_me = inserted_value;

    auto middle = v.begin() + (v.size() / 2);
    auto insertion_begin = v.insert(middle, std::move(insert_me));
    auto insertion_end = insertion_begin + 1;

    assert(insert_me.empty());
    assert(*insertion_begin == inserted_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(insertion_begin, insertion_end, inserted_value) == 1);

    size_t num_initial_elements_before = insertion_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - insertion_end;

    assert(std::count(v.begin(), insertion_begin, initial_value) == static_cast<int>(num_initial_elements_before));
    assert(std::count(insertion_end, v.end(), initial_value) == static_cast<int>(num_initial_elements_after));
  }
}

int main()
{
  test_single_insert_copy();
  test_single_insert_move();

  std::cout << "OK" << std::endl;

  return 0;
}

