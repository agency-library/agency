#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>
#include <cassert>
#include <vector>

template<class ExecutionPolicy1, class ExecutionPolicy2>
void test(ExecutionPolicy1 outer, ExecutionPolicy2 inner)
{
  auto policy = outer(2, inner(3));
  using agent = typename decltype(policy)::execution_agent_type;
  using executor_type = typename decltype(policy)::executor_type;

  {
    // bulk_then with non-void future and no parameters

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    auto f = agency::bulk_then(policy,
      [] __host__ __device__ (agent& self, int& past_arg) -> agency::scope_result<1,int>
      {
        if(self.inner().index() == 0)
        {
          return past_arg;
        }

        return std::ignore;
      },
      fut
    );

    auto result = f.get();

    assert(result == std::vector<int>(2, 7));
  }

  {
    // bulk_then with void future and no parameters

    auto fut = agency::make_ready_future<void>(policy.executor());

    auto f = agency::bulk_then(policy,
      [] __host__ __device__ (agent& self) -> agency::scope_result<1,int>
      {
        if(self.inner().index() == 0)
        {
          return 7;
        }

        return std::ignore;
      },
      fut
    );

    auto result = f.get();

    assert(result == std::vector<int>(2, 7));
  }

  {
    // bulk_then with non-void future and one parameter

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy,
      [] __host__ __device__ (agent& self, int& past_arg, int val) -> agency::scope_result<1,int>
      {
        if(self.inner().index() == 0)
        {
          return past_arg + val;
        }

        return std::ignore;
      },
      fut,
      val
    );

    auto result = f.get();

    assert(result == std::vector<int>(2, 7 + 13));
  }

  {
    // bulk_then with void future and one parameter

    auto fut = agency::make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy,
      [] __host__ __device__ (agent& self, int val) -> agency::scope_result<1,int>
      {
        if(self.inner().index() == 0)
        {
          return val;
        }

        return std::ignore;
      },
      fut,
      val
    );

    auto result = f.get();

    assert(result == std::vector<int>(2, 13));
  }

  {
    // bulk_then with non-void future and one shared parameter
    
    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy,
      [] __host__ __device__ (agent& self, int& past_arg, int& val) -> agency::scope_result<1,int>
      {
        if(self.inner().index() == 0)
        {
          return past_arg + val;
        }

        return std::ignore;
      },
      fut,
      agency::share(val)
    );

    auto result = f.get();

    assert(result == std::vector<int>(2, 7 + 13));
  }


  {
    // bulk_then with void future and one shared parameter
    
    auto fut = agency::make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy,
      [] __host__ __device__ (agent& self, int& val) -> agency::scope_result<1,int>
      {
        if(self.inner().index() == 0)
        {
          return val;
        }

        return std::ignore;
      },
      fut,
      agency::share(val)
    );

    auto result = f.get();

    assert(result == std::vector<int>(2, 13));
  }
}

int main()
{
  using namespace agency;

  test(cuda::par, cuda::con);

  std::cout << "OK" << std::endl;

  return 0;
}

