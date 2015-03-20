#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/bulk_invoke.hpp>

struct hello_world
{
  __device__
  void operator()(agency::cuda::grid_executor_2d::index_type index)
  {
    auto outer = agency::detail::get<0>(index);
    auto inner = agency::detail::get<1>(index);
    printf("Hello world from block {%d,%d}, thread {%d,%d}\n", outer[0], outer[1], inner[0], inner[1]);
  }
};

int main()
{
  agency::cuda::grid_executor_2d ex;

  auto num_blocks = agency::uint2{2,2};
  auto num_threads = agency::uint2{2,2};
  agency::cuda::grid_executor_2d::shape_type shape = {num_blocks, num_threads};

  std::cout << "Testing bulk_invoke on host" << std::endl;
  agency::cuda::bulk_invoke(ex, shape, hello_world());
  cudaDeviceSynchronize();

  return 0;
}

