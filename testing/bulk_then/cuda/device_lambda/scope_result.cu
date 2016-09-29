#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>
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
  auto policy = outer(2, inner(3));
  using agent = typename decltype(policy)::execution_agent_type;
  using executor_type = typename decltype(policy)::executor_type;

  {
    // bulk_then with non-void future and no parameters

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    auto f = agency::bulk_then(policy,
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self, int& past_arg)
      {
        if(self.inner().index() == 0)
        {
          return agency::scope_result<1,int>(past_arg);
        }

        return agency::no_result<int>();
      }),
      fut
    );

    auto result = f.get();

    assert(result == std::vector<int>(2, 7));
  }

  {
    // bulk_then with void future and no parameters

    auto fut = agency::make_ready_future<void>(policy.executor());

    auto f = agency::bulk_then(policy,
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self)
      {
        if(self.inner().index() == 0)
        {
          return agency::scope_result<1,int>(7);
        }

        return agency::no_result<int>();
      }),
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
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self, int& past_arg, int val)
      {
        if(self.inner().index() == 0)
        {
          return agency::scope_result<1,int>(past_arg + val);
        }

        return agency::no_result<int>();
      }),
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
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self, int val)
      {
        if(self.inner().index() == 0)
        {
          return agency::scope_result<1,int>(val);
        }

        return agency::no_result<int>();
      }),
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
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self, int& past_arg, int& val)
      {
        if(self.inner().index() == 0)
        {
          return agency::scope_result<1,int>(past_arg + val);
        }

        return agency::no_result<int>();
      }),
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
      wrap<agency::scope_result<1,int>>([] __device__ (agent& self, int& val)
      {
        if(self.inner().index() == 0)
        {
          return agency::scope_result<1,int>(val);
        }

        return agency::no_result<int>();
      }),
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

