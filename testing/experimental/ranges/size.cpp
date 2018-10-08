#include <vector>
#include <array>
#include <list>
#include <cassert>
#include <iostream>
#include <agency/container/vector.hpp>

#include <agency/experimental/ranges/size.hpp>

struct has_free_function_size {};

constexpr size_t size(has_free_function_size)
{
  return 0;
}

void test()
{
  {
    // test std::vector
    std::vector<int> vec(10);

    assert(agency::experimental::size(vec) == vec.size());
  }

  {
    // test std::array
    std::array<int, 10> arr;

    assert(agency::experimental::size(arr) == arr.size());
  }

  {
    // test std::list
    std::list<int> lst(10);

    assert(agency::experimental::size(lst) == lst.size());
  }

  {
    // test agency::vector
    agency::vector<int> vec(10);

    assert(agency::experimental::size(vec) == vec.size());
  }

  {
    // test has_free_function_size
    has_free_function_size x;

    assert(agency::experimental::size(x) == size(x));
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

