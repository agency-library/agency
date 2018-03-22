#include <agency/execution/execution_policy.hpp>
#include <agency/cuda/execution/execution_policy.hpp>
#include <iostream>

template<class ExecutionPolicy, class Executor>
void test(const ExecutionPolicy& policy, const Executor& ex)
{
  auto p1 = policy.on(ex);
  assert(p1.executor() == ex);
}


int main()
{
  {
    // test seq
    test(agency::seq, agency::sequenced_executor());
  }

  {
    // test par
    test(agency::par, agency::sequenced_executor());
    test(agency::par, agency::parallel_executor());
    test(agency::par, agency::concurrent_executor());
  }

  {
    // test con
    test(agency::con, agency::concurrent_executor());
  }


  {
    // test CUDA-specific policies and executors
    using namespace agency;

    {
      // test par
      test(cuda::par, cuda::parallel_executor());

      test(par, cuda::grid_executor());
      test(cuda::grid(1,1), cuda::grid_executor());
    }

    {
      // test par2d(con2d)
      test(cuda::par2d(size2(1,1), cuda::con2d(size2(1,1))), cuda::grid_executor_2d());
    }

    {
      // test grid
      auto g = agency::cuda::grid(1,1);

      test(g, agency::cuda::grid_executor());
    }
  }

  std::cout << "OK" << std::endl;
}

