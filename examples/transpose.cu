#include <cstdio>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <agency/cuda/execution_policy.hpp>
#include <chrono>

// XXX need to figure out how to make this par(con) select grid_executor_2d automatically
auto grid(agency::size2 num_blocks, agency::size2 num_threads)
  -> decltype(agency::cuda::par(num_blocks, agency::cuda::con(num_threads)).on(agency::cuda::grid_executor_2d{}))
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads)).on(agency::cuda::grid_executor_2d{});
}

using cuda_thread_2d = agency::parallel_group_2d<agency::cuda::concurrent_agent_2d>;


agency::cuda::future<void> async_square_transpose(size_t matrix_dim, float* transposed_matrix, const float* input_matrix)
{
  using namespace agency;

  static constexpr int tile_dim = 32;
  static constexpr int num_rows_per_block = 8;

  size2 dim_grid{matrix_dim/tile_dim, matrix_dim/tile_dim};
  size2 dim_block{tile_dim, num_rows_per_block};

  return cuda::bulk_async(grid(dim_grid, dim_block), [=] __device__ (cuda_thread_2d& self)
  {
    auto idx = tile_dim * self.outer().index() + self.inner().index();

    for(int j = 0; j < tile_dim; j+= num_rows_per_block)
    {
      transposed_matrix[idx[0]*matrix_dim + (idx[1]+j)] = input_matrix[(idx[1]+j)*matrix_dim + idx[0]];
    }
  });
}


template<class Function, class... Args>
double time_invocation(Function f, Args&&... args)
{
  int n = 100;

  // warm up
  f(std::forward<Args>(args)...);

  auto start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < n; i++)
  {
    f(std::forward<Args>(args)...);
  }

  cudaDeviceSynchronize();

  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

  return elapsed.count() / 100;
}


int main(int argc, char **argv)
{
  const int matrix_dim = 1024;

  std::vector<float> input_matrix(matrix_dim * matrix_dim);
  std::vector<float> reference_matrix(matrix_dim * matrix_dim);

  thrust::device_vector<float> d_input_matrix(matrix_dim * matrix_dim);
  thrust::device_vector<float> d_transposed_matrix(matrix_dim * matrix_dim);
    
  // initialize input
  std::iota(input_matrix.begin(), input_matrix.end(), 0);

  // transpose input into reference for error checking
  for(int j = 0; j < matrix_dim; j++)
  {
    for(int i = 0; i < matrix_dim; i++)
    {
      reference_matrix[j*matrix_dim + i] = input_matrix[i*matrix_dim + j];
    }
  }
  
  // copy input to device
  d_input_matrix = input_matrix;

  async_square_transpose(matrix_dim, raw_pointer_cast(d_transposed_matrix.data()), raw_pointer_cast(d_input_matrix.data())).wait();
  assert(reference_matrix == d_transposed_matrix);

  double seconds = time_invocation([&]
  {
    async_square_transpose(matrix_dim, raw_pointer_cast(d_transposed_matrix.data()), raw_pointer_cast(d_input_matrix.data()));
  });

  double gigabytes = double(2 * matrix_dim * matrix_dim * sizeof(float)) / (1 << 30);
  double bandwidth = gigabytes / seconds;

  std::cout << "Matrix Transpose Bandwidth: " << bandwidth << " GB/s" << std::endl;

  return 0;
}

