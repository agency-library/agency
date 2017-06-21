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

  {
    // no parameters

    auto f = agency::bulk_async(policy,
      wrap<int>([] __device__ (agent& self)
    {
      return 7;
    }));

    auto result = f.get();

    auto shape = agent_traits::domain(policy.param()).shape();
    using shape_type = decltype(shape);

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(shape,7) == result);
  }

  {
    // one parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      wrap<int>([] __device__ (agent& self, int val)
    {
      return val;
    }),
    val);

    auto result = f.get();

    auto shape = agent_traits::domain(policy.param()).shape();
    using shape_type = decltype(shape);

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(shape,val) == result);
  }

  {
    // one shared parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      wrap<int>([] __device__ (agent& self, int& val)
    {
      return val;
    }),
    agency::share(val));

    auto result = f.get();

    auto shape = agent_traits::domain(policy.param()).shape();
    using shape_type = decltype(shape);

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(shape,val) == result);
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

