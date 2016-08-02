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

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = typename agency::executor_traits<executor_type>::template container<int>;

    auto shape = agent_traits::domain(policy.param()).shape();

    using container_shape_t = typename container_type::shape_type;
    auto container_shape = agency::detail::shape_cast<container_shape_t>(shape);

    assert(container_type(container_shape,7) == result);
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

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = typename agency::executor_traits<executor_type>::template container<int>;

    auto shape = agent_traits::domain(policy.param()).shape();

    using container_shape_t = typename container_type::shape_type;
    auto container_shape = agency::detail::shape_cast<container_shape_t>(shape);

    assert(container_type(container_shape,val) == result);
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

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = typename agency::executor_traits<executor_type>::template container<int>;

    auto shape = agent_traits::domain(policy.param()).shape();

    using container_shape_t = typename container_type::shape_type;
    auto container_shape = agency::detail::shape_cast<container_shape_t>(shape);

    assert(container_type(container_shape,val) == result);
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

