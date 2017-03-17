#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>

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
  for(size_t i = 0; i < v.size(); ++i)
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

  {
    // test converting copy construction
    flatten_view<const std::vector<std::vector<int>>> flattened2 = flattened;

    std::vector<int> expected_values(flattened.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(size_t i = 0; i < flattened2.size(); ++i)
    {
      assert(flattened2[i] == expected_values[i]);
    }
  }

  {
    // test iterator traits

    using iterator = decltype(flattened.begin());

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
    assert(flattened.begin() == flattened.begin());
  }

  {
    // test iterator inequality
    assert(!(flattened.begin() != flattened.begin()));
  }

  {
    // test sentinel equality
    assert(flattened.end() == flattened.end());
  }

  {
    // test iterator/sentinel inequality
    assert(flattened.begin() != flattened.end());
  }

  {
    // test iterator/sentinel difference
    size_t expected_size = flattened.end() - flattened.begin();
    assert(expected_size == flattened.size());
  }

  {
    // test iterator pre-increment
    auto iter = flattened.begin();

    auto iter2 = ++iter;
    assert(iter - flattened.begin() == 1);
    assert(iter2 - iter == 0);
    assert(iter2 == iter);
  }

  {
    // test iterator post-increment
    auto iter = flattened.begin();

    auto iter2 = iter++;
    assert(iter - flattened.begin() == 1);
    assert(iter - iter2 == 1);
    assert(iter2 != iter);
  }

  {
    // test iterator pre-decrement
    auto iter = flattened.end();

    auto iter2 = --iter;
    assert(flattened.end() - iter == 1);
    assert(iter2 - iter == 0);
    assert(iter2 == iter);
  }

  {
    // test iterator post-decrement
    auto iter = flattened.end();

    auto iter2 = iter--;
    assert(flattened.end() - iter == 1);
    assert(iter2 - iter == 1);
    assert(iter2 != iter);
  }

  {
    // test iterator plus-assign
    auto iter = flattened.begin();

    iter += 2;
    assert(iter - flattened.begin() == 2);
  }

  {
    // test iterator minus-assign
    auto iter = flattened.end();

    iter -= 2;
    assert(flattened.end() - iter == 2);
  }

  {
    // test iterator plus
    auto iter = flattened.begin() + 2;

    assert(iter - flattened.begin() == 2);
  }

  {
    // test iterator minus
    auto iter = flattened.end() - 2;

    assert(flattened.end() - iter == 2);
  }

  {
    // test iterator dereference
    
    std::vector<int> expected_values(flattened.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(auto iter = flattened.begin(); iter != flattened.end(); ++iter)
    {
      auto i = iter - flattened.begin();

      assert(*iter == expected_values[i]);
    }
  }

  {
    // test iterator bracket
    
    std::vector<int> expected_values(flattened.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    for(size_t i = 0; i < flattened.size(); ++i)
    {
      auto iter = flattened.begin();

      assert(iter[i] == expected_values[i]);
    }
  }

  {
    // test assign through iterator dereference

    std::vector<std::vector<int>> v;
    v.emplace_back(std::vector<int>(3, 0));
    v.emplace_back(std::vector<int>(2, 0));
    v.emplace_back(std::vector<int>(5, 0));

    auto flattened2 = flatten(v);

    assert(flattened2.size() == flattened.size());

    auto src = flattened.begin();
    for(auto dst = flattened2.begin(); dst != flattened2.end(); ++dst, ++src)
    {
      *dst = *src;
    }


    std::vector<int> expected_values(flattened.size());
    std::iota(expected_values.begin(), expected_values.end(), 0);

    assert(std::equal(expected_values.begin(), expected_values.end(), flattened2.begin()));
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

