#include <agency/execution_policy.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <atomic>

__managed__ int counter;

template<class ExecutionPolicy>
void test(ExecutionPolicy policy)
{
  using agent = typename ExecutionPolicy::execution_agent_type;
  using agent_traits = agency::execution_agent_traits<agent>;

  {
    // bulk_invoke with no parameters

    counter = 0;

    agency::bulk_invoke(policy, [] __device__ (agent& self)
    {
      atomicAdd(&counter, 1);
    });

    size_t num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents);
  }

  {
    // bulk_invoke with one parameter

    int val = 13;

    counter = 0;

    agency::bulk_invoke(policy,
      [] __device__ (agent& self, int val)
      {
        atomicAdd(&counter, val);
      },
      val
    );

    size_t num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 13);
  }

  {
    // bulk_invoke with one shared parameter

    int val = 13;

    counter = 0;

    agency::bulk_invoke(policy,
      [] __device__ (agent& self, int& val)
      {
        atomicAdd(&counter, val);
      },
      agency::share(val)
    );

    size_t num_agents = agent_traits::domain(policy.param()).size();

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

