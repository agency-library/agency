#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>

__managed__ int result;

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

struct functor
{
  int *result;

  template<class Index>
  __device__
  void operator()(const Index&, int& past, int& outer_shared, int& inner_shared, int& inner_inner_shared)
  {
    atomicAdd(result, past + outer_shared + inner_shared + inner_inner_shared);
  }
};

int main()
{
  using namespace agency;

  using outer_executor_type = cuda::this_thread::parallel_executor;
  using inner_executor_type = cuda::grid_executor;

  int num_devices = 0;
  cudaError_t error = cudaGetDeviceCount(&num_devices);
  if(error)
  {
    std::string what("CUDA error after cudaGetDeviceCount(): ");
    what += std::string(cudaGetErrorString(error));
    throw std::runtime_error(what);
  }

  {
    // test executor_array then_execute()
    using executor_type = executor_array<inner_executor_type, outer_executor_type>;
    using shape_type = executor_shape_t<executor_type>;
    using index_type = executor_index_t<executor_type>;
    using allocator_type = executor_allocator_t<executor_type,int>;
    using container_type = bulk_result<int,shape_type,allocator_type>;

    executor_type exec(num_devices);

    for(size_t i = 0; i < exec.size(); ++i)
    {
      exec[i].device(i);
    }

    shape_type shape = exec.make_shape(exec.size(),{2,2});

    auto past = agency::make_ready_future<int>(exec, 13);

    auto f = exec.bulk_then_execute([=] __host__ __device__ (const index_type& idx, int& past, container_type& results, int& outer_shared, int& inner_shared, int& inner_inner_shared)
    {
      printf("hello from agent %d %d %d\n", (int)agency::get<0>(idx), (int)agency::get<1>(idx), (int)agency::get<2>(idx));
      results[idx] = past + outer_shared + inner_shared + inner_inner_shared;
    },
    shape,
    past,
    [=] __host__ __device__ { return container_type(shape); },
    make_7(),
    make_42(),
    make_1()
    );

    auto results = f.get();

    assert(results.size() == agency::detail::index_space_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7 + 42 + 1; }));
  }

  {
    // test flattened executor_array
    using executor_array_type = executor_array<inner_executor_type, outer_executor_type>;
    using executor_type = flattened_executor<executor_array_type>;

    using shape_type = executor_shape_t<executor_type>;
    using index_type = executor_index_t<executor_type>;
    using allocator_type = executor_allocator_t<executor_type,int>;
    using container_type = bulk_result<int,shape_type,allocator_type>;

    executor_array_type exec_array(num_devices);

    for(size_t i = 0; i < exec_array.size(); ++i)
    {
      exec_array[i].device(i);
    }

    executor_type exec{exec_array};

    shape_type shape{exec_array.size(), 2};

    auto ready = agency::make_ready_future<void>(exec);

    auto f = exec.bulk_then_execute([] __host__ __device__ (const index_type& idx, container_type& results, int& outer_shared, int& inner_shared)
    {
      results[idx] = 13 + outer_shared + inner_shared;
    },
    shape,
    ready,
    [=] __host__ __device__ { return container_type(shape); },
    make_7(),
    make_42()
    );

    auto results = f.get();

    assert(results.size() == agency::detail::index_space_size(shape));
    assert(std::all_of(results.begin(), results.end(), [](int x){ return x == 13 + 7 + 42; }));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

