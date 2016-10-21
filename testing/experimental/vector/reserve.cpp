#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <agency/experimental/vector.hpp>

void test_allocating_reserve()
{
  using namespace agency::experimental;

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

    assert(std::count(v.begin(), v.end(), 13) == num_initial_elements);

    size_t reservation = 100;

    v.reserve(reservation);

    assert(v.capacity() >= reservation);
    assert(v.size() == num_initial_elements);
    assert(!v.empty());
    assert(std::count(v.begin(), v.end(), 13) == num_initial_elements);
  }
}

int main()
{
  test_allocating_reserve();

  std::cout << "OK" << std::endl;

  return 0;
}

