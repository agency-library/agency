#include <agency/execution_policy.hpp>
#include <atomic>

template<class ExecutionPolicy>
void test()
{
  using execution_policy_type = ExecutionPolicy;

  {
    // bulk_invoke with no parameters

    execution_policy_type policy;
    auto exec = policy.executor();

    std::atomic<int> counter{0};

    auto f = agency::bulk_async(policy(10),
      [&](typename execution_policy_type::execution_agent_type& self)
    {
      ++counter;
    });

    f.wait();

    assert(counter == 10);
  }

  {
    // bulk_invoke with one parameter
    
    execution_policy_type policy;

    int val = 13;

    std::atomic<int> counter{0};

    auto f = agency::bulk_async(policy(10),
      [&](typename execution_policy_type::execution_agent_type& self, int val)
      {
        counter += val;
      },
      val
    );

    f.wait();

    assert(counter == 10 * 13);
  }

  {
    // bulk_invoke with one shared parameter
    
    execution_policy_type policy;

    int val = 13;

    std::atomic<int> counter{0};

    auto f = agency::bulk_async(policy(10),
      [&](typename execution_policy_type::execution_agent_type& self, int& val)
      {
        counter += val;
      },
      agency::share(val)
    );

    f.wait();

    assert(counter == 10 * 13);
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

