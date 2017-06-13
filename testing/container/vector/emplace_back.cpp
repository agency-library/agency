#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <agency/container/vector.hpp>

void test_emplace_back()
{
  using namespace agency;

  {
    // test no arg emplace_back into empty vector
    std::vector<int> emplaced_value;

    vector<std::vector<int>> v;

    auto& result = v.emplace_back();

    assert(emplaced_value == result);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test copying emplace_back into empty vector
    std::vector<int> emplaced_value(1,7);

    vector<std::vector<int>> v;

    auto& result = v.emplace_back(emplaced_value);

    assert(!emplaced_value.empty());
    assert(emplaced_value == result);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test moving emplace_back into empty vector
    std::vector<int> emplaced_value(1,7);

    vector<std::vector<int>> v;

    auto emplace_me = emplaced_value;
    auto& result = v.emplace_back(std::move(emplace_me));

    assert(emplace_me.empty());
    assert(emplaced_value == result);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test emplace_back from args into empty vector
    std::vector<int> emplaced_value(1,7);

    vector<std::vector<int>> v;

    auto& result = v.emplace_back(1,7);

    assert(emplaced_value == result);
    assert(v.size() == 1);
    assert(std::count(v.begin(), v.end(), emplaced_value) == 1);
  }

  {
    // test no arg emplace_back into non-empty vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value;

    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    auto& result = v.emplace_back();

    assert(result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.end()-1, v.end(), emplaced_value) == 1);
    assert(std::count(v.begin(), v.end()-1, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test copying emplace_back into non-empty vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;
    auto& result = v.emplace_back(emplace_me);

    assert(!emplace_me.empty());
    assert(result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.end()-1, v.end(), emplaced_value) == 1);
    assert(std::count(v.begin(), v.end()-1, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test moving emplace_back into non-empty vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    std::vector<int> emplace_me = emplaced_value;
    auto& result = v.emplace_back(std::move(emplace_me));

    assert(emplace_me.empty());
    assert(result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.end()-1, v.end(), emplaced_value) == 1);
    assert(std::count(v.begin(), v.end()-1, initial_value) == static_cast<int>(num_initial_elements));
  }

  {
    // test emplace_back from args into non-empty vector
    std::vector<int> initial_value(1,13);
    std::vector<int> emplaced_value(1,7);
    
    size_t num_initial_elements = 10;

    vector<std::vector<int>> v(num_initial_elements, initial_value);

    auto& result = v.emplace_back(1,7);

    assert(result == emplaced_value);
    assert(v.size() == num_initial_elements + 1);
    assert(std::count(v.end()-1, v.end(), emplaced_value) == 1);
    assert(std::count(v.begin(), v.end()-1, initial_value) == static_cast<int>(num_initial_elements));
  }
}

int main()
{
  test_emplace_back();

  std::cout << "OK" << std::endl;

  return 0;
}

