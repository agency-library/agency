#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>

template<class ExecutionPolicy>
void test(ExecutionPolicy policy)
{
  using agent = typename ExecutionPolicy::execution_agent_type;
  using agent_traits = agency::execution_agent_traits<agent>;

  {
    // bulk_async with no parameters

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent& self)
    {
      return 7;
    });

    auto result = f.get();

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::executor_container_t<executor_type,int>;

    auto shape = agent_traits::domain(policy.param()).shape();

    using container_shape_t = typename container_type::shape_type;
    auto container_shape = agency::detail::shape_cast<container_shape_t>(shape);

    assert(container_type(container_shape,7) == result);
  }

  {
    // bulk_async with one parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent& self, int val)
    {
      return val;
    },
    val);

    auto result = f.get();

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::executor_container_t<executor_type,int>;

    auto shape = agent_traits::domain(policy.param()).shape();

    using container_shape_t = typename container_type::shape_type;
    auto container_shape = agency::detail::shape_cast<container_shape_t>(shape);

    assert(container_type(container_shape,val) == result);
  }

  {
    // bulk_async with one shared parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent& self, int& val)
    {
      return val;
    },
    agency::share(val));

    auto result = f.get();

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::executor_container_t<executor_type,int>;

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
  test(con(10));

  test(par(10, con(10)));

  std::cout << "OK" << std::endl;

  return 0;
}

