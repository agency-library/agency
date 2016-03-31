#include <agency/execution_policy.hpp>

template<class ExecutionPolicy>
void test()
{
  using execution_policy_type = ExecutionPolicy;

  {
    // bulk_async with no parameters

    execution_policy_type policy;
    auto exec = policy.executor();

    auto f = agency::bulk_async(policy(10),
      [](typename execution_policy_type::execution_agent_type& self) -> agency::single_result<int>
    {
      if(self.index() == 0)
      {
        return 7;
      }

      return std::ignore;
    });

    auto result = f.get();

    assert(result == 7);
  }

  {
    // bulk_async with one parameter
    
    execution_policy_type policy;

    int val = 13;

    auto f = agency::bulk_async(policy(10),
      [](typename execution_policy_type::execution_agent_type& self, int val) -> agency::single_result<int>
    {
      if(self.index() == 0)
      {
        return val;
      }

      return std::ignore;
    },
    val);

    auto result = f.get();

    assert(result == 13);
  }

  {
    // bulk_invoke with one shared parameter
    
    execution_policy_type policy;

    int val = 13;

    auto f = agency::bulk_async(policy(10),
      [](typename execution_policy_type::execution_agent_type& self, int& val) -> agency::single_result<int>
    {
      if(self.index() == 0)
      {
        return val;
      }

      return std::ignore;
    },
    agency::share<0>(val));

    auto result = f.get();

    assert(result == 13);
  }
}

int main()
{
  test<agency::sequential_execution_policy>();
  test<agency::concurrent_execution_policy>();
  test<agency::parallel_execution_policy>();

  std::cout << "OK" << std::endl;

  return 0;
}

