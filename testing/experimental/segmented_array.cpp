#include <agency/agency.hpp>
#include <agency/experimental/segmented_array.hpp>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>

void test()
{
  using namespace agency::experimental;
  using namespace agency;

  {
    // test default constructor
    segmented_array<int> array;

    assert(array.size() == 0);
  }

  {
    // test construction with full segments

    size_t segment_size = 5;

    std::vector<allocator<int>> allocators(10);

    segmented_array<int> array(segment_size * allocators.size(), 13, allocators);

    std::vector<int> expected_values(segment_size * allocators.size(), 13);

    assert(array.size() == segment_size * allocators.size());
    assert(std::equal(array.begin(), array.end(), expected_values.begin()));
  }

  {
    // test construction with one partial segment

    size_t segment_size = 5;

    std::vector<allocator<int>> allocators(10);

    size_t num_elements = segment_size * (allocators.size() - 1) + (segment_size - 1);

    segmented_array<int> array(num_elements, 13, allocators);

    std::vector<int> expected_values(num_elements, 13);

    assert(array.size() == num_elements);
    assert(std::equal(array.begin(), array.end(), expected_values.begin()));
  }

  {
    // test construction with empty segments

    size_t num_elements = 4;
    size_t num_allocators = 10;

    std::vector<allocator<int>> allocators(num_allocators);

    segmented_array<int> array(num_elements, 13, allocators);

    std::vector<int> expected_values(num_elements, 13);

    assert(array.size() == num_elements);
    assert(std::equal(array.begin(), array.end(), expected_values.begin()));
  }

  {
    // test copy construction

    size_t segment_size = 5;

    std::vector<allocator<int>> allocators(10);

    segmented_array<int> array(segment_size * allocators.size(), 13, allocators);

    segmented_array<int> copy = array;

    std::vector<int> expected_values(segment_size * allocators.size(), 13);

    assert(copy.size() == segment_size * allocators.size());
    assert(std::equal(copy.begin(), copy.end(), expected_values.begin()));
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

