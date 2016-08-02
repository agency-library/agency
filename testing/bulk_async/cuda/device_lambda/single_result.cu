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


template<class ExecutionPolicy>
void test(ExecutionPolicy policy)
{
  using agent = typename ExecutionPolicy::execution_agent_type;

  {
    // no parameters

    auto f = agency::bulk_async(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self)
    {
      if(self.elect())
      {
        return agency::single_result<int>(7);
      }

      return agency::no_result<int>();
    }));

    auto result = f.get();

    assert(result == 7);
  }

  {
    // one parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self, int val)
    {
      if(self.elect())
      {
        return agency::single_result<int>(val);
      }

      return agency::no_result<int>();
    }),
    val);

    auto result = f.get();

    assert(result == 13);
  }

  {
    // one shared parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self, int& val)
    {
      if(self.elect())
      {
        return agency::single_result<int>(val);
      }

      return agency::no_result<int>();
    }),
    agency::share(val));

    auto result = f.get();

    assert(result == 13);
  }
}

int main()
{
  using namespace agency::cuda;

  test(par(10));
  test(con(10));

  test(par(10, con(10)));

  std::cout << "OK" << std::endl;

  return 0;
}

