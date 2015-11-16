#include <agency/cuda/grid_executor.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>

struct return_second_parameter
{
  template<class Index, class Arg, class... Args>
  __device__
  Arg operator()(const Index&, Arg& shared_arg, Args&... shared_args)
  {
    return shared_arg;
  }
};

int main()
{
  {
    using executor_type = agency::cuda::grid_executor;
    executor_type exec;

    auto ready = agency::cuda::make_ready_future();

    executor_type::shape_type shape{100,256};

    auto f = exec.then_execute([] __device__ (executor_type::index_type idx, int& outer, int& inner)
    {
      return outer + inner;
    },
    [](executor_type::shape_type shape)
    {
      return executor_type::container<int>(shape);
    },
    shape,
    ready,
    agency::detail::make_factory(7),
    agency::detail::make_factory(13)
    );

    auto result = f.get();

    assert(std::all_of(result.begin(), result.end(), [](int x) { return x == 20; }));
  }

  {
    using executor_type = agency::flattened_executor<agency::cuda::grid_executor>;
    executor_type exec;

    auto ready = agency::cuda::make_ready_future();

    executor_type::shape_type shape = 1;

    auto f = exec.then_execute(
      return_second_parameter(),
      [](executor_type::shape_type shape)
      {
        return executor_type::container<int>(shape);
      },
      shape,
      ready,
      agency::detail::make_factory(7)
    );

    auto result = f.get();

    assert(std::all_of(result.begin(), result.end(), [](int x) { return x == 7; }));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

