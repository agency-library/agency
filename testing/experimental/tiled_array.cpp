#include <agency/agency.hpp>
#include <agency/experimental/tiled_array.hpp>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>

void test()
{
  using namespace agency;
  using namespace agency::experimental;

  {
    // test default constructor
    tiled_array<int> array;

    assert(array.size() == 0);
  }

  {
    // test construction with full segments

    size_t tile_size = 5;

    std::vector<allocator<int>> allocators(8);

    tiled_array<int> array(tile_size * allocators.size(), 13, allocators);

    std::vector<int> expected_values(tile_size * allocators.size(), 13);

    assert(array.size() == tile_size * allocators.size());
    assert(std::equal(array.begin(), array.end(), expected_values.begin()));
  }

  {
    // test construction with one partial segment

    size_t tile_size = 5;

    std::vector<allocator<int>> allocators(8);

    size_t num_elements = tile_size * (allocators.size() - 1) + (tile_size - 1);

    tiled_array<int> array(num_elements, 13, allocators);

    std::vector<int> expected_values(num_elements, 13);

    assert(array.size() == num_elements);
    assert(std::equal(array.begin(), array.end(), expected_values.begin()));
  }

  {
    // test construction with empty segments

    size_t num_elements = 4;
    size_t num_allocators = 8;

    std::vector<allocator<int>> allocators(num_allocators);

    tiled_array<int> array(num_elements, 13, allocators);

    std::vector<int> expected_values(num_elements, 13);

    assert(array.size() == num_elements);
    assert(std::equal(array.begin(), array.end(), expected_values.begin()));
  }

  {
    // test copy construction

    size_t tile_size = 5;

    std::vector<allocator<int>> allocators(8);

    tiled_array<int> array(tile_size * allocators.size(), 13, allocators);

    tiled_array<int> copy = array;

    std::vector<int> expected_values(tile_size * allocators.size(), 13);

    assert(copy.size() == tile_size * allocators.size());
    assert(std::equal(copy.begin(), copy.end(), expected_values.begin()));
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

