#include <agency/agency.hpp>
#include <agency/cuda.hpp>
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

    auto f = agency::bulk_async(policy, [] __host__ __device__ (agent&)
    {
#ifdef __CUDA_ARCH__
      atomicAdd(&counter, 1);
#endif
    });

    f.wait();

    int num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents);
  }

  {
    // bulk_invoke with one parameter

    int val = 13;

    counter = 0;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent&, int val)
      {
        // WAR unused parameter warning
        (void)val;

#ifdef __CUDA_ARCH__
        atomicAdd(&counter, val);
#endif
      },
      val
    );

    f.wait();

    int num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 13);
  }

  {
    // bulk_invoke with one shared parameter

    int val = 13;

    counter = 0;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent&, int& val)
      {
        // WAR unused parameter warning
        (void)val;

#ifdef __CUDA_ARCH__
        atomicAdd(&counter, val);
#endif
      },
      agency::share(val)
    );

    f.wait();

    int num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 13);
  }
}

int main()
{
  using namespace agency::cuda;

  test(con(10));
  test(par(10));

  test(par(10, con(10)));

  std::cout << "OK" << std::endl;

  return 0;
}

