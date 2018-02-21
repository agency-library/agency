#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <agency/container/vector.hpp>

void test_range_erase()
{
  using namespace agency;

  {
    // test erase empty vector

    vector<int> v;

    auto iterator = v.erase(v.begin(), v.end());

    assert(iterator == v.end());
    assert(iterator == v.begin());
    assert(v.empty());
  }

  {
    // test erase entire non-empty vector

    vector<int> v(10);

    auto iterator = v.erase(v.begin(), v.end());

    assert(iterator == v.end());
    assert(iterator == v.begin());
    assert(v.empty());
  }

  {
    // test erase middle of non-empty vector

    vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);

    size_t num_elements_before_erase = v.size() / 2;
    size_t num_elements_after_erase = v.size() - num_elements_before_erase;

    auto erase_first = v.begin() + num_elements_before_erase;
    auto erase_last = v.end() - num_elements_after_erase;

    size_t num_elements_erased = erase_last - erase_first;

    auto result = v.erase(erase_first, erase_last);

    assert(result == v.begin() + num_elements_before_erase);
    assert(v.size() == num_elements_before_erase + num_elements_after_erase);

    std::vector<int> elements_before_erase(num_elements_before_erase);
    std::iota(elements_before_erase.begin(), elements_before_erase.end(), 0);
    assert(std::equal(elements_before_erase.begin(), elements_before_erase.end(), v.begin()));

    std::vector<int> elements_after_erase(num_elements_after_erase);
    std::iota(elements_after_erase.begin(), elements_after_erase.end(), num_elements_before_erase + num_elements_erased);
    assert(std::equal(elements_after_erase.begin(), elements_after_erase.end(), result));
  }
}

template<class ExecutionPolicy>
void test_range_erase(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test erase empty vector

    vector<int> v;

    auto iterator = v.erase(policy, v.begin(), v.end());

    assert(iterator == v.end());
    assert(iterator == v.begin());
    assert(v.empty());
  }

  {
    // test erase entire non-empty vector

    vector<int> v(10);

    auto iterator = v.erase(policy, v.begin(), v.end());

    assert(iterator == v.end());
    assert(iterator == v.begin());
    assert(v.empty());
  }

  {
    // test erase middle of non-empty vector

    vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);

    size_t num_elements_before_erase = v.size() / 2;
    size_t num_elements_after_erase = v.size() - num_elements_before_erase;

    auto erase_first = v.begin() + num_elements_before_erase;
    auto erase_last = v.end() - num_elements_after_erase;

    size_t num_elements_erased = erase_last - erase_first;

    auto result = v.erase(policy, erase_first, erase_last);

    assert(result == v.begin() + num_elements_before_erase);
    assert(v.size() == num_elements_before_erase + num_elements_after_erase);

    std::vector<int> elements_before_erase(num_elements_before_erase);
    std::iota(elements_before_erase.begin(), elements_before_erase.end(), 0);
    assert(std::equal(elements_before_erase.begin(), elements_before_erase.end(), v.begin()));

    std::vector<int> elements_after_erase(num_elements_after_erase);
    std::iota(elements_after_erase.begin(), elements_after_erase.end(), num_elements_before_erase + num_elements_erased);
    assert(std::equal(elements_after_erase.begin(), elements_after_erase.end(), result));
  }
}

int main()
{
  test_range_erase();
  test_range_erase(agency::seq);
  test_range_erase(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

