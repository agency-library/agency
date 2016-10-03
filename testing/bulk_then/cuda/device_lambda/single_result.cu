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
  using agent_traits = agency::execution_agent_traits<agent>;
  using executor_type = typename ExecutionPolicy::executor_type;

  {
    // non-void future and no parameters

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    auto f = agency::bulk_then(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self, int& past_arg)
      {
        if(self.elect())
        {
          return agency::single_result<int>(past_arg);
        }

        return agency::no_result<int>();
      }),
      fut
    );

    auto result = f.get();

    assert(result == 7);
  }

  {
    // void future and no parameters

    auto fut = agency::make_ready_future<void>(policy.executor());

    auto f = agency::bulk_then(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self)
      {
        if(self.elect())
        {
          return agency::single_result<int>(7);
        }

        return std::ignore;
      }),
      fut
    );

    auto result = f.get();

    assert(result == 7);
  }

  {
    // non-void future and one parameter

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self, int& past_arg, int val)
      {
        if(self.elect())
        {
          return agency::single_result<int>(past_arg + val);
        }

        return agency::no_result<int>();
      }),
      fut,
      val
    );

    auto result = f.get();

    assert(result == 7 + 13);
  }

  {
    // void future and one parameter
    
    auto fut = agency::make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self, int val)
      {
        if(self.elect())
        {
          return agency::single_result<int>(val);
        }

        return agency::no_result<int>();
      }),
      fut,
      val
    );

    auto result = f.get();

    assert(result == 13);
  }

  {
    // bulk_then with non-void future and one shared parameter

    auto fut = agency::make_ready_future<int>(policy.executor(), 7);

    int val = 13;

    auto f = agency::bulk_then(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self, int& past_arg, int& val)
      {
        if(self.elect())
        {
          return agency::single_result<int>(past_arg + val);
        }

        return agency::no_result<int>();
      }),
      fut,
      agency::share(val)
    );

    auto result = f.get();

    assert(result == 7 + 13);
  }

  {
    // bulk_then with void future and one shared parameter

    auto fut = agency::make_ready_future<void>(policy.executor());

    int val = 13;

    auto f = agency::bulk_then(policy,
      wrap<agency::single_result<int>>([] __device__ (agent& self, int& val)
      {
        if(self.elect())
        {
          return agency::single_result<int>(val);
        }

        return agency::no_result<int>();
      }),
      fut,
      agency::share(val)
    );

    auto result = f.get();

    assert(result == 13);
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

