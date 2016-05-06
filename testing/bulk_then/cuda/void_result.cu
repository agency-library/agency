#include <agency/bulk_then.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <iostream>

__managed__ unsigned int counter;

template<class ExecutionPolicy>
void test()
{
  using execution_policy_type = ExecutionPolicy;
  using executor_type = typename ExecutionPolicy::executor_type;

  {
    // bulk_then with non-void future and no parameters

    execution_policy_type policy;
    auto exec = policy.executor();

    counter = 0;

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<int>(policy.executor(), 7);

    auto f = agency::bulk_then(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self, int& past_arg)
      {
#ifdef __CUDA_ARCH__
        atomicAdd(&counter, past_arg);
#endif
      },
      fut
    );

    f.wait();

    assert(counter == 10 * 7);
  }

  {
    // bulk_then with void future and no parameters

    execution_policy_type policy;
    auto exec = policy.executor();

    counter = 0;

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<void>(policy.executor());

    auto f = agency::bulk_then(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self)
      {
#ifdef __CUDA_ARCH__
        atomicAdd(&counter, 1);
#endif
      },
      fut
    );

    f.wait();

    assert(counter == 10);
  }

  {
    // bulk_then with non-void future and one parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    counter = 0;

    auto f = agency::bulk_then(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self, int& past_arg, int val)
      {
#ifdef __CUDA_ARCH__
        atomicAdd(&counter, past_arg + val);
#endif
      },
      fut,
      val
    );

    f.wait();

    assert(counter == 10 * (7 + 13));
  }

  {
    // bulk_then with void future and one parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    counter = 0;

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self, int val)
      {
#ifdef __CUDA_ARCH__
        atomicAdd(&counter, val);
#endif
      },
      fut,
      val
    );

    f.wait();

    assert(counter == 10 * 13);
  }

  {
    // bulk_then with non-void future and one shared parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    counter = 0;

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self, int& past_arg, int& val)
      {
#ifdef __CUDA_ARCH__
        atomicAdd(&counter, past_arg + val);
#endif
      },
      fut,
      agency::share(val)
    );

    f.wait();

    assert(counter == 10 * (7 + 13));
  }

  {
    // bulk_then with void future and one shared parameter
    
    execution_policy_type policy;
    auto exec = policy.executor();

    counter = 0;

    auto fut = agency::executor_traits<executor_type>::template make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy(10),
      [] __host__ __device__ (typename execution_policy_type::execution_agent_type& self, int& val)
      {
#ifdef __CUDA_ARCH__
        atomicAdd(&counter, val);
#endif
      },
      fut,
      agency::share(val)
    );

    f.wait();

    assert(counter == 10 * 13);
  }
}

int main()
{
  test<agency::cuda::parallel_execution_policy>();

  std::cout << "OK" << std::endl;

  return 0;
}

