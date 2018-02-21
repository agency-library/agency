#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <list>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>


template<class Container>
void test_range_constructor()
{
  using namespace agency;

  using value_type = typename Container::value_type;

  size_t num_elements_to_insert = 5;
  Container items(num_elements_to_insert);
  std::iota(items.begin(), items.end(), 0);

  vector<value_type> v(items.begin(), items.end());

  assert(v.size() == num_elements_to_insert);
  assert(std::equal(v.begin(), v.end(), items.begin()));
}


template<class Container, class ExecutionPolicy>
void test_range_constructor(ExecutionPolicy policy)
{
  using namespace agency;

  using value_type = typename Container::value_type;

  size_t num_elements_to_insert = 5;
  Container items(num_elements_to_insert);
  std::iota(items.begin(), items.end(), 0);

  vector<value_type> v(items.begin(), items.end());

  assert(v.size() == num_elements_to_insert);
  assert(std::equal(v.begin(), v.end(), items.begin()));
}


int main()
{
  {
    // test construction from std::vector
    test_range_constructor<std::vector<int>>();
    test_range_constructor<std::vector<int>>(agency::par);
  }

  {
    // test construction from std::list, which has iterators which do not parallelize 
    test_range_constructor<std::list<int>>();
  }

  std::cout << "OK" << std::endl;

  return 0;
}

