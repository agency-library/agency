#include <agency/agency.hpp>
#include <iostream>

template<class ExecutionPolicy>
void test()
{
  using execution_policy_type = ExecutionPolicy;

  {
    // bulk_invoke with no parameters
    
    execution_policy_type policy;

    auto result = agency::bulk_invoke(policy(10),
      [](typename execution_policy_type::execution_agent_type&)
    {
      return 7;
    });

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::vector<int, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(10,7) == result);
  }

  {
    // bulk_invoke with one parameter
    
    execution_policy_type policy;

    int val = 13;

    auto result = agency::bulk_invoke(policy(10),
      [](typename execution_policy_type::execution_agent_type&, int val)
    {
      return val;
    },
    val);

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::vector<int, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(10,val) == result);
  }

  {
    // bulk_invoke with one shared parameter
    
    execution_policy_type policy;

    int val = 13;

    auto result = agency::bulk_invoke(policy(10),
      [](typename execution_policy_type::execution_agent_type&, int& val)
    {
      return val;
    },
    agency::share(val));

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::vector<int, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(10,val) == result);
  }
}

int main()
{
  test<agency::sequenced_execution_policy>();
  test<agency::concurrent_execution_policy>();
  test<agency::parallel_execution_policy>();

  std::cout << "OK" << std::endl;

  return 0;
}

