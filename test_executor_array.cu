#include <cassert>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <agency/executor_array.hpp>
#include <agency/cuda/parallel_executor.hpp>

struct make_7
{
  __host__ __device__
  int operator()() const
  {
    return 7;
  }
};

struct make_42
{
  __host__ __device__
  int operator()() const
  {
    return 42;
  }
};

struct make_1
{
  __host__ __device__
  int operator()() const
  {
    return 1;
  }
};

int main()
{
  using namespace agency;

  using outer_executor_type = cuda::this_thread::parallel_executor;
  using inner_executor_type = cuda::grid_executor;

  {
    // test executor_array async_execute()
    using executor_type = executor_array<inner_executor_type, outer_executor_type>;
    using traits = agency::executor_traits<executor_type>;
    using shape_type = typename traits::shape_type;
    using index_type = typename traits::index_type;

    executor_type exec(2);

    for(size_t i = 0; i < 2; ++i)
    {
      exec[i].gpu(i);
    }

    auto shape = exec.make_shape(2,{2,2});

    auto past = traits::make_ready_future<int>(exec, 13);

    auto f = exec.then_execute([=] __device__ (const index_type& idx, int& past, int& outer_shared, int& inner_shared, int& inner_inner_shared)
    {
      printf("hello from agent %d %d %d\n", (int)agency::detail::get<0>(idx), (int)agency::detail::get<1>(idx), (int)agency::detail::get<2>(idx));
      return past + outer_shared + inner_shared + inner_inner_shared;
    },
    [](shape_type shape)
    {
      return traits::container<int>(shape);
    },
    shape,
    past,
    make_7(),
    make_42(),
    make_1());

    auto results = f.get();

    assert(results.size() == agency::detail::shape_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7 + 42 + 1; }));
  }

  {
    // test executor_array async_execute()
    using executor_type = executor_array<inner_executor_type, outer_executor_type>;
    using traits = agency::executor_traits<executor_type>;
    using shape_type = typename traits::shape_type;
    using index_type = typename traits::index_type;

    executor_type exec(2);

    for(size_t i = 0; i < 2; ++i)
    {
      exec[i].gpu(i);
    }

    auto shape = exec.make_shape(2,{2,2});

    auto f = exec.async_execute([] __device__ (const index_type& idx, int& outer_shared, int& inner_shared, int& inner_inner_shared)
    {
      return 13 + outer_shared + inner_shared + inner_inner_shared;
    },
    [](shape_type shape)
    {
      return traits::container<int>(shape);
    },
    shape,
    make_7(),
    make_42(),
    make_1());

    auto results = f.get();

    assert(results.size() == agency::detail::shape_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7 + 42 + 1; }));
  }

  {
    // test flattened executor_array
    using executor_array_type = executor_array<inner_executor_type, outer_executor_type>;
    using executor_type = flattened_executor<executor_array_type>;

    using traits = agency::executor_traits<executor_type>;
    using shape_type = typename traits::shape_type;
    using index_type = typename traits::index_type;

    executor_array_type exec_array(2);

    for(size_t i = 0; i < 2; ++i)
    {
      exec_array[i].gpu(i);
    }

    executor_type exec{exec_array};

    shape_type shape{2, 2};

    auto ready = traits::make_ready_future<void>(exec);

    auto f = exec.then_execute([] __device__ (const index_type& idx, int& outer_shared, int& inner_shared)
    {
      return 13 + outer_shared + inner_shared;
    },
    [](shape_type shape)
    {
      return traits::container<int>(shape);
    },
    shape,
    ready,
    make_7(),
    make_42());

    auto results = f.get();

    assert(results.size() == agency::detail::shape_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7 + 42; }));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

