#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <agency/container/vector.hpp>

void test_single_emplace_copy()
{
  using namespace agency;

  {
    // test single emplace copy into empty vector
    std::vector<int> emplaced_value(1,7);

    vector<std::vector<int>> v;

    std::vector<int> emplace_me = emplaced_value;
    auto result = v.emplace(v.begin(), emplace_me);

    assert(!emplace_me.empty());
    assert(result == v.begin());
    assert(*result == emplaced_value);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test single emplace copy at the beginning of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;
    auto result = v.emplace(v.begin(), emplace_me);

    assert(!emplace_me.empty());
    assert(result == v.begin());
    assert(*result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test single emplace copy at the end of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;
    auto result = v.emplace(v.end(), emplace_me);

    assert(!emplace_me.empty());
    assert(result == v.end() - 1);
    assert(*result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(result, result + 1, emplaced_value) == 1);
    assert(std::count(v.begin(), result, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test single emplace copy in the middle of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);

    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;

    auto middle = v.begin() + (v.size() / 2);
    auto emplace_begin = v.emplace(middle, emplace_me);
    auto emplace_end = emplace_begin + 1;

    assert(!emplace_me.empty());
    assert(*emplace_begin == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(emplace_begin, emplace_end, emplaced_value) == 1);

    size_t num_initial_elements_before = emplace_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - emplace_end;

    assert(std::count(v.begin(), emplace_begin, initial_value) == static_cast<int>(num_initial_elements_before));
    assert(std::count(emplace_end, v.end(), initial_value) == static_cast<int>(num_initial_elements_after));
  }
}

void test_single_emplace_move()
{
  using namespace agency;

  {
    // test single emplace move into empty vector
    std::vector<int> emplaced_value(1,7);

    vector<std::vector<int>> v;

    std::vector<int> emplace_me = emplaced_value;
    auto result = v.emplace(v.begin(), std::move(emplace_me));

    assert(emplace_me.empty());
    assert(result == v.begin());
    assert(*result == emplaced_value);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test single emplace move at the beginning of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;
    auto result = v.emplace(v.begin(), std::move(emplace_me));

    assert(emplace_me.empty());
    assert(result == v.begin());
    assert(*result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test single emplace move at the end of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;
    auto result = v.emplace(v.end(), std::move(emplace_me));

    assert(emplace_me.empty());
    assert(result == v.end() - 1);
    assert(*result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(result, result + 1, emplaced_value) == 1);
    assert(std::count(v.begin(), result, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test single emplace move in the middle of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);

    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;

    auto middle = v.begin() + (v.size() / 2);
    auto emplace_begin = v.emplace(middle, std::move(emplace_me));
    auto emplace_end = emplace_begin + 1;

    assert(emplace_me.empty());
    assert(*emplace_begin == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(emplace_begin, emplace_end, emplaced_value) == 1);

    size_t num_initial_elements_before = emplace_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - emplace_end;

    assert(std::count(v.begin(), emplace_begin, initial_value) == static_cast<int>(num_initial_elements_before));
    assert(std::count(emplace_end, v.end(), initial_value) == static_cast<int>(num_initial_elements_after));
  }
}


void test_single_emplace_args()
{
  using namespace agency;

  {
    // test single emplace move into empty vector
    std::vector<int> emplaced_value(1,7);

    vector<std::vector<int>> v;

    auto result = v.emplace(v.begin(), 1, 7);

    assert(result == v.begin());
    assert(*result == emplaced_value);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test single emplace move at the beginning of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    auto result = v.emplace(v.begin(), 1, 7);

    assert(result == v.begin());
    assert(*result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test single emplace move at the end of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    auto result = v.emplace(v.end(), 1, 7);

    assert(result == v.end() - 1);
    assert(*result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(result, result + 1, emplaced_value) == 1);
    assert(std::count(v.begin(), result, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test single emplace move in the middle of vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);

    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    auto middle = v.begin() + (v.size() / 2);
    auto emplace_begin = v.emplace(middle, 1, 7);
    auto emplace_end = emplace_begin + 1;

    assert(*emplace_begin == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(emplace_begin, emplace_end, emplaced_value) == 1);

    size_t num_initial_elements_before = emplace_begin - v.begin();
    size_t num_initial_elements_after  = v.end() - emplace_end;

    assert(std::count(v.begin(), emplace_begin, initial_value) == static_cast<int>(num_initial_elements_before));
    assert(std::count(emplace_end, v.end(), initial_value) == static_cast<int>(num_initial_elements_after));
  }
}

int main()
{
  test_single_emplace_copy();
  test_single_emplace_move();
  test_single_emplace_args();

  std::cout << "OK" << std::endl;

  return 0;
}

