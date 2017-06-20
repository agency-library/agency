#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <agency/experimental/ndarray.hpp>
#include <iostream>

template<class ExecutionPolicy>
void test(ExecutionPolicy policy)
{
  using agent = typename ExecutionPolicy::execution_agent_type;
  using agent_traits = agency::execution_agent_traits<agent>;

  {
    // bulk_async with no parameters

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent&)
    {
      return 7;
    });

    auto result = f.get();

    auto shape = agent_traits::domain(policy.param()).shape();
    using shape_type = decltype(shape);

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(shape,7) == result);
  }

  {
    // bulk_async with one parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent&, int val)
    {
      return val;
    },
    val);

    auto result = f.get();

    auto shape = agent_traits::domain(policy.param()).shape();
    using shape_type = decltype(shape);

    using executor_type = typename ExecutionPolicy::executor_type;
    using container_type = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<executor_type,int>>;

    assert(container_type(shape,val) == result);
  }

  {
    // bulk_async with one shared parameter

    int val = 13;

    auto f = agency::bulk_async(policy,
      [] __host__ __device__ (agent&, int& val)
    {
      return val;
    },
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
  test(con(10));

  test(par(10, con(10)));

  std::cout << "OK" << std::endl;

  return 0;
}

