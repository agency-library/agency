#include <agency/agency.hpp>
#include <agency/cuda.hpp>

struct hello_world
{
  __device__
  void operator()(agency::cuda::grid_executor_2d::index_type index)
  {
    auto outer = agency::get<0>(index);
    auto inner = agency::get<1>(index);
    printf("Hello world from block {%d,%d}, thread {%d,%d}\n", outer[0], outer[1], inner[0], inner[1]);
  }
};

int main()
{
  agency::cuda::grid_executor_2d ex;

  auto num_blocks = agency::uint2{2,2};
  auto num_threads = agency::uint2{2,2};
  agency::cuda::grid_executor_2d::shape_type shape = {num_blocks, num_threads};

  std::cout << "Testing execute on host" << std::endl;
  agency::detail::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(ex, hello_world(), shape);

  return 0;
}

