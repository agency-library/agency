#include <agency/agency.hpp>

template<class ExecutionPolicy>
void test()
{
  using execution_policy_type = ExecutionPolicy;
  using executor_type = typename ExecutionPolicy::executor_type;

  {
    // bulk_then with non-void future and no parameters

    execution_policy_type policy;
    auto exec = policy.executor();

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<int>(policy.executor(), 7);

    auto f = agency::bulk_then(policy(10),
      [](typename execution_policy_type::execution_agent_type& self, int& past_arg) -> agency::single_result<int>
      {
        if(self.index() == 0)
        {
          return past_arg;
        }

        return std::ignore;
      },
      fut
    );

    auto result = f.get();

    assert(result == 7);
  }

  {
    // bulk_then with void future and no parameters

    execution_policy_type policy;
    auto exec = policy.executor();

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<void>(policy.executor());

    auto f = agency::bulk_then(policy(10),
      [](typename execution_policy_type::execution_agent_type& self) -> agency::single_result<int>
      {
        if(self.index() == 0)
        {
          return 7;
        }

        return std::ignore;
      },
      fut
    );

    auto result = f.get();

    assert(result == 7);
  }

  {
    // bulk_then with non-void future and one parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy(10),
      [](typename execution_policy_type::execution_agent_type& self, int& past_arg, int val) -> agency::single_result<int>
      {
        if(self.index() == 0)
        {
          return past_arg + val;
        }

        return std::ignore;
      },
      fut,
      val
    );

    auto result = f.get();

    assert(result == 7 + 13);
  }

  {
    // bulk_then with void future and one parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy(10),
      [](typename execution_policy_type::execution_agent_type& self, int val) -> agency::single_result<int>
      {
        if(self.index() == 0)
        {
          return val;
        }

        return std::ignore;
      },
      fut,
      val
    );

    auto result = f.get();

    assert(result == 13);
  }

  {
    // bulk_then with non-void future and one shared parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy(10),
      [](typename execution_policy_type::execution_agent_type& self, int& past_arg, int& val) -> agency::single_result<int>
      {
        if(self.index() == 0)
        {
          return past_arg + val;
        }

        return std::ignore;
      },
      fut,
      agency::share(val)
    );

    auto result = f.get();

    assert(result == 7 + 13);
  }

  {
    // bulk_then with void future and one shared parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy(10),
      [](typename execution_policy_type::execution_agent_type& self, int& val) -> agency::single_result<int>
      {
        if(self.index() == 0)
        {
          return val;
        }

        return std::ignore;
      },
      fut,
      agency::share(val)
    );

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

