#include <agency/cuda/grid_executor.hpp>
#include <agency/flattened_executor.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>

// XXX we have to define these as functors in the global scope because of nvbug 1703880
struct return_7
{
  template<class Index>
  __device__
  int operator()(const Index&)
  {
    return 7;
  }
};

struct return_sum_of_last_two_parameters
{
  template<class Index>
  __device__
  int operator()(const Index&, int& arg1, int& arg2)
  {
    return arg1 + arg2;
  }
};

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
    // no past parameter, two shared parameters
    using executor_type = agency::cuda::grid_executor;
    using traits = agency::executor_traits<executor_type>;
    executor_type exec;

    auto ready = traits::make_ready_future<void>(exec);

    executor_type::shape_type shape{100,256};

    auto f = traits::then_execute(exec,
      [] __device__ (executor_type::index_type idx, int& outer, int& inner)
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
    // one past parameter, one shared parameter
    using executor_type = agency::flattened_executor<agency::cuda::grid_executor>;
    using traits = agency::executor_traits<executor_type>;
    executor_type exec;

    auto ready = traits::make_ready_future<int>(exec, 13);

    executor_type::shape_type shape = 100;

    auto f = traits::then_execute(exec,
      return_sum_of_last_two_parameters(),
      [](executor_type::shape_type shape)
      {
        return executor_type::container<int>(shape);
      },
      shape,
      ready,
      agency::detail::make_factory(7)
    );

    auto result = f.get();

    assert(std::all_of(result.begin(), result.end(), [](int x) { return x == 20; }));
  }

  {
    // no past parameter, shared parameter
    using executor_type = agency::flattened_executor<agency::cuda::grid_executor>;
    using traits = agency::executor_traits<executor_type>;
    executor_type exec;

    auto ready = traits::make_ready_future<void>(exec);

    executor_type::shape_type shape = 100;

    auto f = traits::then_execute(
      exec,
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

