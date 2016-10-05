#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>

#include <agency/experimental/ranges/flatten.hpp>

void test()
{
  using namespace agency::experimental;

  // create 4 vectors, each with a different number of elements
  std::vector<std::vector<int>> v;
  v.emplace_back(std::vector<int>(4));
  v.emplace_back(std::vector<int>(1));
  v.emplace_back(std::vector<int>(3));
  v.emplace_back(std::vector<int>(2));

  // initialize vectors with ascending integers
  int init = 0;
  for(int i = 0; i < v.size(); ++i)
  {
    std::iota(v[i].begin(), v[i].end(), init);
    init = v[i].back() + 1;
  }

  auto flattened = flatten(v);

  {
    // test .size()
    
    size_t expected_size = 0;
    for(auto& segment : v)
    {
      expected_size += segment.size();
    }

    assert(flattened.size() == expected_size);
  }

  {
    // test operator[]
    
    std::vector<int> expected_values(flattened.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(size_t i = 0; i < flattened.size(); ++i)
    {
      assert(flattened[i] == expected_values[i]);
    }
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

