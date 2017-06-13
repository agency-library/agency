#include <iostream>
#include <cassert>
#include <algorithm>
#include <agency/container/vector.hpp>
#include <agency/execution/execution_policy.hpp>

void test_clear()
{
  using namespace agency;

  {
    // test clear empty vector
    
    vector<int> v;

    v.clear();

    assert(v.empty());
  }

  {
    // test clear vector
    
    size_t num_elements = 10;

    vector<int> v(num_elements, 13);

    v.clear();

    assert(v.empty());
  }
}

template<class ExecutionPolicy>
void test_clear(ExecutionPolicy policy)
{
  using namespace agency;

  {
    // test clear empty vector
    
    vector<int> v;

    v.clear(policy);

    assert(v.empty());
  }

  {
    // test clear vector
    
    size_t num_elements = 10;

    vector<int> v(num_elements, 13);

    v.clear(policy);

    assert(v.empty());
  }
}

int main()
{
  test_clear();
  test_clear(agency::seq);
  test_clear(agency::par);

  std::cout << "OK" << std::endl;

  return 0;
}

