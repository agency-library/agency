#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <cassert>
#include <vector>

template<class ExecutionPolicy1, class ExecutionPolicy2>
void test(ExecutionPolicy1 outer, ExecutionPolicy2 inner)
{
  {
    // bulk_async with no parameters

    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent& self) -> agency::scope_result<1,int>
    {
      if(self.inner().index() == 0)
      {
        return 7;
      }

      return std::ignore;
    });

    auto result = f.get();

    assert(result == std::vector<int>(2, 7));
  }

  {
    // bulk_async with one parameter
    
    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    int val = 13;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent& self, int val) -> agency::scope_result<1,int>
    {
      if(self.inner().index() == 0)
      {
        return val;
      }

      return std::ignore;
    },
    val);

    auto result = f.get();

    assert(result == std::vector<int>(2, 13));
  }

  {
    // bulk_async with one shared parameter
    
    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    int val = 13;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent& self, int& val) -> agency::scope_result<1,int>
    {
      if(self.inner().index() == 0)
      {
        return val;
      }

      return std::ignore;
    },
    agency::share(val));

    auto result = f.get();

    assert(result == std::vector<int>(2, 13));
  }
}

int main()
{
  using namespace agency::cuda;

  test(par, con);

  std::cout << "OK" << std::endl;

  return 0;
}

