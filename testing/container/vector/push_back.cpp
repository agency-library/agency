#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <agency/container/vector.hpp>

void test_push_back()
{
  using namespace agency;

  {
    // test copying push_back into empty vector
    std::vector<int> copied_value(1,7);

    vector<std::vector<int>> v;

    v.push_back(copied_value);
    auto& result = v.back();

    assert(!copied_value.empty());
    assert(copied_value == result);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), copied_value) == 1);
  }

  {
    // test moving push_back into empty vector
    std::vector<int> moved_value(1,7);

    vector<std::vector<int>> v;

    auto move_me = moved_value;
    v.push_back(std::move(move_me));
    auto& result = v.back();

    assert(move_me.empty());
    assert(moved_value == result);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), moved_value) == 1);
  }

  {
    // test copying push_back into non-empty vector
    std::vector<int> initial_value(1,13);
    std::vector<int> copied_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> copy_me = copied_value;
    v.push_back(copy_me);
    auto& result = v.back();

    assert(!copy_me.empty());
    assert(result == copied_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.end()-1, v.end(), copied_value) == 1);
    assert(std::count(v.begin(), v.end()-1, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test moving push_back into non-empty vector
    std::vector<int> initial_value(1,13);
    std::vector<int> moved_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> move_me = moved_value;
    v.push_back(std::move(move_me));
    auto& result = v.back();

    assert(move_me.empty());
    assert(result == moved_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.end()-1, v.end(), moved_value) == 1);
    assert(std::count(v.begin(), v.end()-1, initial_value) == static_cast<int>(num_initial_elements));
  }
}

int main()
{
  test_push_back();

  std::cout << "OK" << std::endl;

  return 0;
}

