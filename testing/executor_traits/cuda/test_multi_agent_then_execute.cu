#include <agency/cuda/grid_executor.hpp>
#include <algorithm>
#include <iostream>


__managed__ int increment_me;

struct increment_and_return_void
{
  template<class Index>
  __device__
  void operator()(const Index&, int& past)
  {
    atomicAdd(&increment_me, past);
  }
};


template<class Container>
struct container_factory
{
  template<class Shape>
  __host__ __device__
  Container operator()(const Shape& shape)
  {
    return Container(shape);
  }
};


template<class Executor>
void test()
{
  using executor_type = Executor;

  using traits = agency::executor_traits<executor_type>;

  {
    // then_execute returning user-specified container
    
    executor_type exec;

    typename executor_type::shape_type shape{10,10};

    auto past = traits::template make_ready_future<int>(exec, 13);

    using container_type = typename executor_type::template container<int>;

    using index_type = typename executor_type::index_type;

    auto fut = traits::then_execute(exec, [] __host__ __device__ (index_type idx, int& past)
    {
      return past;
    },
    container_factory<container_type>{},
    shape,
    past);

    auto got = fut.get();

    assert(got == std::vector<int>(got.size(), 13));
  }

  {
    // then_execute returning default container
    
    executor_type exec;

    typename executor_type::shape_type shape{10,10};

    auto past = traits::template make_ready_future<int>(exec, 13);

    using index_type = typename traits::index_type;

    auto fut = traits::then_execute(exec, [] __host__ __device__ (index_type idx, int& past)
    {
      return past;
    },
    shape,
    past);

    auto result = fut.get();

    std::vector<int> ref(result.size(), 13);
    assert(std::equal(ref.begin(), ref.end(), result.begin()));
  }

  {
    // then_execute returning void
    
    executor_type exec;

    typename executor_type::shape_type shape{10,10};

    auto past = traits::template make_ready_future<int>(exec, 13);

    increment_me = 0;

    using index_type = typename traits::index_type;

    // XXX don't use a __device__ lambda here because we can't reliably compute its return type
    auto fut = traits::then_execute(exec, increment_and_return_void(), shape, past);

    fut.wait();

    assert(increment_me == shape[0] * shape[1] * 13);
  }
}

int main()
{
  // a completely empty executor
  test<agency::cuda::grid_executor>();

  std::cout << "OK" << std::endl;

  return 0;
}

