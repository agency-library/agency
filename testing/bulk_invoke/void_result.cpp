#include <agency/agency.hpp>
#include <atomic>
#include <cassert>

template<class ExecutionPolicy>
void test()
{
  using execution_policy_type = ExecutionPolicy;

  {
    // bulk_invoke with no parameters

    execution_policy_type policy;

    std::atomic<int> counter{0};

    agency::bulk_invoke(policy(10),
      [&](typename execution_policy_type::execution_agent_type&)
    {
      ++counter;
    });

    assert(counter == 10);
  }

  {
    // bulk_invoke with one parameter
    
    execution_policy_type policy;

    int val = 13;

    std::atomic<int> counter{0};

    agency::bulk_invoke(policy(10),
      [&](typename execution_policy_type::execution_agent_type&, int val)
      {
        counter += val;
      },
      val
    );

    assert(counter == 10 * 13);
  }

  {
    // bulk_invoke with one shared parameter
    
    execution_policy_type policy;

    int val = 13;

    std::atomic<int> counter{0};

    agency::bulk_invoke(policy(10),
      [&](typename execution_policy_type::execution_agent_type&, int& val)
      {
        counter += val;
      },
      agency::share(val)
    );

    assert(counter == 10 * 13);
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

