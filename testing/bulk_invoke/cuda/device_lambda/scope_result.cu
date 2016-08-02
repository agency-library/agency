#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <cassert>
#include <vector>


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


template<class ExecutionPolicy1, class ExecutionPolicy2>
void test(ExecutionPolicy1 outer, ExecutionPolicy2 inner)
{
  {
    // no parameters

    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    auto result = agency::bulk_invoke(policy,
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self)
    {
      if(self.inner().index() == 0)
      {
        return agency::scope_result<1,int>(7);
      }

      return agency::no_result<int>();
    }));

    assert(result == std::vector<int>(2, 7));
  }

  {
    // one parameter
    
    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    int val = 13;

    auto result = agency::bulk_invoke(policy,
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self, int val)
    {
      if(self.inner().index() == 0)
      {
        return agency::scope_result<1,int>(val);
      }

      return agency::no_result<int>();
    }),
    val);

    assert(result == std::vector<int>(2, 13));
  }

  {
    // one shared parameter
    
    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    int val = 13;

    auto result = agency::bulk_invoke(policy,
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self, int& val)
    {
      if(self.inner().index() == 0)
      {
        return agency::scope_result<1,int>(val);
      }

      return agency::no_result<int>();
    }),
    agency::share(val));

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

