#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/experimental/vector.hpp>

void test_fill_construct()
{
  using namespace agency::experimental;

  {
    // test fill construct empty vector
    vector<int> v(0, 13);

    assert(v.begin() == v.end());
    assert(v.cbegin() == v.cend());
    assert(v.size() == 0);
    assert(v.empty());
  }

  {
    size_t num_elements = 10;

    // test fill construct vector
    vector<int> v(num_elements, 13);

    assert(v.end() - v.begin() == num_elements);
    assert(v.cend() - v.cbegin() == num_elements);
    assert(v.size() == num_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == 10);
  }
}

void test_fill_insert()
{
  using namespace agency::experimental;

  {
    size_t num_initial_elements = 10;

    // test fill insert at the beginning of vector
    vector<int> v(num_initial_elements, 13);

    size_t num_elements_to_insert = 5;
    auto result = v.insert(v.begin(), num_elements_to_insert, 7);

    assert(result == v.begin());
    assert(v.size() == num_initial_elements + num_elements_to_insert);
    assert(std::count(result, result + num_elements_to_insert, 7) == num_elements_to_insert);
    assert(std::count(result + num_elements_to_insert, v.end(), 13) == num_initial_elements);
  }
}

int main()
{
  test_fill_construct();
  test_fill_insert();

  std::cout << "OK" << std::endl;

  return 0;
}

