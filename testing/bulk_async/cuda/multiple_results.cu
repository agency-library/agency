#include <agency/execution_policy.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <iostream>

template<class ExecutionPolicy>
void test()
{
  using execution_policy_type = ExecutionPolicy;

  {
    // bulk_async with no parameters
    
    execution_policy_type policy;

    auto f = agency::bulk_async(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self)
    {
      return 7;
    });

    auto result = f.get();

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = typename agency::executor_traits<executor_type>::template container<int>;

    assert(container_type(10,7) == result);
  }

  {
    // bulk_async with one parameter
    
    execution_policy_type policy;

    int val = 13;

    auto f = agency::bulk_async(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self, int val)
    {
      return val;
    },
    val);

    auto result = f.get();

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = typename agency::executor_traits<executor_type>::template container<int>;

    assert(container_type(10,val) == result);
  }

  {
    // bulk_async with one shared parameter
    
    execution_policy_type policy;

    int val = 13;

    auto f = agency::bulk_async(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self, int& val)
    {
      return val;
    },
    agency::share(val));

    auto result = f.get();

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = typename agency::executor_traits<executor_type>::template container<int>;

    assert(container_type(10,val) == result);
  }
}

int main()
{
  test<agency::cuda::parallel_execution_policy>();

  std::cout << "OK" << std::endl;

  return 0;
}

