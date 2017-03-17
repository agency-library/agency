#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>

#include <agency/experimental/ranges/untile.hpp>

void test()
{
  using namespace agency::experimental;

  size_t tile_size = 5;
  size_t last_tile_size = 3;

  // create 4 vectors, each with tile_size number of elements except the last
  std::vector<std::vector<int>> v;
  v.emplace_back(std::vector<int>(tile_size));
  v.emplace_back(std::vector<int>(tile_size));
  v.emplace_back(std::vector<int>(tile_size));
  v.emplace_back(std::vector<int>(last_tile_size));

  // initialize vectors with ascending integers
  int init = 0;
  for(auto& tile : v)
  {
    std::iota(tile.begin(), tile.end(), init);
    init = tile.back() + 1;
  }

  auto untiled = untile(tile_size, v);

  {
    // test .size()
    
    size_t expected_size = 0;
    for(auto& tile : v)
    {
      expected_size += tile.size();
    }

    assert(untiled.size() == expected_size);
  }

  {
    // test operator[]
    
    std::vector<int> expected_values(untiled.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(size_t i = 0; i < untiled.size(); ++i)
    {
      assert(untiled[i] == expected_values[i]);
    }
  }

  {
    // test converting copy construction
    small_untiled_view<const std::vector<std::vector<int>>> untiled2 = untiled;

    std::vector<int> expected_values(untiled.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(size_t i = 0; i < untiled2.size(); ++i)
    {
      assert(untiled2[i] == expected_values[i]);
    }
  }

  {
    // test iterator traits

    using iterator = decltype(untiled.begin());

    using value_type = std::iterator_traits<iterator>::value_type;
    using reference = std::iterator_traits<iterator>::reference;
    using pointer = std::iterator_traits<iterator>::pointer;
    using difference_type = std::iterator_traits<iterator>::difference_type;
    using iterator_category = std::iterator_traits<iterator>::iterator_category;

    static_assert(std::is_same<value_type, int>::value,                                    "value_type should be int");
    static_assert(std::is_same<reference, int&>::value,                                    "reference should be int&");
    static_assert(std::is_same<pointer, int*>::value,                                      "pointer should be int*");
    static_assert(std::is_same<difference_type, std::ptrdiff_t>::value,                    "difference_type should be std::ptrdiff_t");
    static_assert(std::is_same<iterator_category, std::random_access_iterator_tag>::value, "iterator_category should be std::random_access_iterator_tag");
  }

  {
    // test iterator equality
    assert(untiled.begin() == untiled.begin());
  }

  {
    // test iterator inequality
    assert(!(untiled.begin() != untiled.begin()));
  }

  {
    // test sentinel equality
    assert(untiled.end() == untiled.end());
  }

  {
    // test iterator/sentinel inequality
    assert(untiled.begin() != untiled.end());
  }

  {
    // test iterator/sentinel difference
    ptrdiff_t expected_difference = untiled.size();
    assert(untiled.end() - untiled.begin() == expected_difference);
  }

  {
    // test iterator pre-increment
    auto iter = untiled.begin();

    auto iter2 = ++iter;
    assert(iter - untiled.begin() == 1);
    assert(iter2 - iter == 0);
    assert(iter2 == iter);
  }

  {
    // test iterator post-increment
    auto iter = untiled.begin();

    auto iter2 = iter++;
    assert(iter - untiled.begin() == 1);
    assert(iter - iter2 == 1);
    assert(iter2 != iter);
  }

  {
    // test iterator pre-decrement
    auto iter = untiled.end();

    auto iter2 = --iter;
    assert(untiled.end() - iter == 1);
    assert(iter2 - iter == 0);
    assert(iter2 == iter);
  }

  {
    // test iterator post-decrement
    auto iter = untiled.end();

    auto iter2 = iter--;
    assert(untiled.end() - iter == 1);
    assert(iter2 - iter == 1);
    assert(iter2 != iter);
  }

  {
    // test iterator plus-assign
    auto iter = untiled.begin();

    iter += 2;
    assert(iter - untiled.begin() == 2);
  }

  {
    // test iterator minus-assign
    auto iter = untiled.end();

    iter -= 2;
    assert(untiled.end() - iter == 2);
  }

  {
    // test iterator plus
    auto iter = untiled.begin() + 2;

    assert(iter - untiled.begin() == 2);
  }

  {
    // test iterator minus
    auto iter = untiled.end() - 2;

    assert(untiled.end() - iter == 2);
  }

  {
    // test iterator dereference
    
    std::vector<int> expected_values(untiled.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(auto iter = untiled.begin(); iter != untiled.end(); ++iter)
    {
      auto i = iter - untiled.begin();

      assert(*iter == expected_values[i]);
    }
  }

  {
    // test iterator bracket
    
    std::vector<int> expected_values(untiled.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(size_t i = 0; i < untiled.size(); ++i)
    {
      auto iter = untiled.begin();

      assert(iter[i] == expected_values[i]);
    }
  }

  {
    // test assign through iterator dereference

    size_t tile_size2 = 7;
    size_t last_tile_size2 = 4;
    std::vector<std::vector<int>> v;
    v.emplace_back(std::vector<int>(tile_size2, 0));
    v.emplace_back(std::vector<int>(tile_size2, 0));
    v.emplace_back(std::vector<int>(last_tile_size2, 0));

    auto untiled2 = untile(tile_size2, v);

    assert(untiled2.size() == untiled.size());

    auto src = untiled.begin();
    for(auto dst = untiled2.begin(); dst != untiled2.end(); ++dst, ++src)
    {
      *dst = *src;
    }


    std::vector<int> expected_values(untiled.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    assert(std::equal(expected_values.begin(), expected_values.end(), untiled2.begin()));
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

