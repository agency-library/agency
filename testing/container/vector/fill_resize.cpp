#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>

void test_enlarging_fill_resize()
{
  using namespace agency;

  size_t old_size = 10;

  vector<int> v(old_size, 13);

  size_t new_size = old_size + 5;
  v.resize(new_size, 7);

  assert(v.size() == new_size);
  assert(std::count(v.begin(), v.begin() + old_size, 13) == static_cast<int>(old_size));
  assert(std::count(v.begin() + old_size, v.end(), 7) == static_cast<int>(new_size - old_size));
}

template<class ExecutionPolicy>
void test_enlarging_fill_resize(ExecutionPolicy policy)
{
  using namespace agency;

  size_t old_size = 10;

  vector<int> v(old_size, 13);

  size_t new_size = old_size + 5;
  v.resize(policy, new_size, 7);

  assert(v.size() == new_size);
  assert(std::count(v.begin(), v.begin() + old_size, 13) == static_cast<int>(old_size));
  assert(std::count(v.begin() + old_size, v.end(), 7) == static_cast<int>(new_size - old_size));
}

void test_shrinking_fill_resize()
{
  using namespace agency;

  size_t old_size = 10;

  vector<int> v(old_size, 13);

  size_t new_size = old_size - 5;
  v.resize(new_size, 7);

  assert(v.size() == new_size);
  assert(std::count(v.begin(), v.end(), 13) == static_cast<int>(new_size));
  assert(std::count(v.begin(), v.end(), 7)  == 0);
}

template<class ExecutionPolicy>
void test_shrinking_fill_resize(ExecutionPolicy policy)
{
  using namespace agency;

  size_t old_size = 10;

  vector<int> v(old_size, 13);

  size_t new_size = old_size - 5;
  v.resize(policy, new_size, 7);

  assert(v.size() == new_size);
  assert(std::count(v.begin(), v.end(), 13) == static_cast<int>(new_size));
  assert(std::count(v.begin(), v.end(), 7)  == 0);
}

int main()
{
  test_enlarging_fill_resize();
  test_enlarging_fill_resize(agency::seq);
  test_enlarging_fill_resize(agency::par);

  test_shrinking_fill_resize();
  test_shrinking_fill_resize(agency::seq);
  test_shrinking_fill_resize(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

