#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/coordinate.hpp>

void test_fill_constructor()
{
  using namespace agency;

  {
    // test fill construct 1D point
    constexpr size_t num_elements = 1;
    
    point<int,num_elements> p(13);

    ptrdiff_t expected_difference = num_elements;

    assert(std::count(p.begin(), p.end(), 13) == expected_difference);
  }

  {
    // test fill construct 2D point
    constexpr size_t num_elements = 2;
    
    point<int,num_elements> p(13);

    ptrdiff_t expected_difference = num_elements;

    assert(std::count(p.begin(), p.end(), 13) == expected_difference);
  }

  {
    // test fill construct 3D point
    constexpr size_t num_elements = 3;
    
    point<int,num_elements> p(13);

    ptrdiff_t expected_difference = num_elements;

    assert(std::count(p.begin(), p.end(), 13) == expected_difference);
  }
}

int main()
{
  test_fill_constructor();

  std::cout << "OK" << std::endl;

  return 0;
}

