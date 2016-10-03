#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>


template<class T, class Function>
struct device_lambda_wrapper
{
  Function f;

  template<class... Args>
  __device__
  T operator()(Args&&... args)
  {
    return f(std::forward<Args>(args)...);
  }
};


template<class T, class Function>
device_lambda_wrapper<T,Function> wrap(Function f)
{
  return device_lambda_wrapper<T,Function>{f};
}


__managed__ unsigned int counter;


template<class ExecutionPolicy>
void test(ExecutionPolicy policy)
{
  using agent = typename ExecutionPolicy::execution_agent_type;
  using agent_traits = agency::execution_agent_traits<agent>;
  using executor_type = typename ExecutionPolicy::executor_type;

  {
    // non-void future and no parameters

    counter = 0;

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    auto f = agency::bulk_then(policy,
      [] __device__ (agent& self, int& past_arg)
      {
        atomicAdd(&counter, past_arg);
      },
      fut
    );

    f.wait();

    auto num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 7);
  }

  {
    // void future and no parameters

    counter = 0;

    auto fut = agency::make_ready_future<void>(policy.executor());

    auto f = agency::bulk_then(policy,
      [] __device__ (agent& self)
      {
        atomicAdd(&counter, 1);
      },
      fut
    );

    f.wait();

    auto num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents);
  }

  {
    // non-void future and one parameter
    
    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    counter = 0;

    auto f = agency::bulk_then(policy,
      [] __device__ (agent& self, int& past_arg, int val)
      {
        atomicAdd(&counter, past_arg + val);
      },
      fut,
      val
    );

    f.wait();

    auto num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * (7 + 13));
  }

  {
    // void future and one parameter
    
    counter = 0;

    auto fut = agency::make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy,
      [] __device__ (agent& self, int val)
      {
        atomicAdd(&counter, val);
      },
      fut,
      val
    );

    f.wait();

    auto num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 13);
  }

  {
    // non-void future and one shared parameter
    
    counter = 0;

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy,
      [] __device__ (agent& self, int& past_arg, int& val)
      {
        atomicAdd(&counter, past_arg + val);
      },
      fut,
      agency::share(val)
    );

    f.wait();

    auto num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * (7 + 13));
  }

  {
    // void future and one shared parameter
    
    counter = 0;

    auto fut = agency::make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy,
      [] __device__ (agent& self, int& val)
      {
        atomicAdd(&counter, val);
      },
      fut,
      agency::share(val)
    );

    f.wait();

    auto num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 13);
  }
}

int main()
{
  using namespace agency::cuda;

  test(par(10));

  test(par(10, con(10)));

  std::cout << "OK" << std::endl;

  return 0;
}

