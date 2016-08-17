#include <cassert>
#include <numeric>
#include <chrono>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>


agency::cuda::future<void> async_square_transpose(size_t matrix_dim, float* transposed_matrix, const float* input_matrix)
{
  using namespace agency;

  int tile_dim = 32;
  int num_rows_per_block = 8;

  size2 outer_shape{matrix_dim/tile_dim, matrix_dim/tile_dim};
  size2 inner_shape{tile_dim, num_rows_per_block};

  return bulk_async(cuda::grid(outer_shape, inner_shape), 
    [=] __host__ __device__ (cuda::grid_agent_2d& self)
    {
      auto idx = tile_dim * self.outer().index() + self.inner().index();

      for(int j = 0; j < tile_dim; j += num_rows_per_block)
      {
        transposed_matrix[idx[0]*matrix_dim + (idx[1]+j)] = input_matrix[(idx[1]+j)*matrix_dim + idx[0]];
      }
    }
  );
}


template<class Function, class... Args>
double time_invocation(Function f, Args&&... args)
{
  size_t num_trials = 100;

  // warm up
  f(std::forward<Args>(args)...);

  auto start = std::chrono::high_resolution_clock::now();

  for(size_t i = 0; i < num_trials; i++)
  {
    f(std::forward<Args>(args)...);
  }

  cudaDeviceSynchronize();

  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

  return elapsed.count() / num_trials;
}


int main()
{
  const int matrix_dim = 1024;

  thrust::device_vector<float> input_matrix(matrix_dim * matrix_dim);
  thrust::device_vector<float> transposed_matrix(matrix_dim * matrix_dim);
    
  // initialize input
  std::iota(input_matrix.begin(), input_matrix.end(), 0);

  // transpose sequentially to compute a reference
  thrust::host_vector<float> reference_matrix = input_matrix;
  for(int i = 0; i < matrix_dim - 1; ++i)
  {
    for(int j = i + 1; j < matrix_dim; ++j)
    {
      std::swap(reference_matrix[i*matrix_dim + j], reference_matrix[j*matrix_dim + i]);
    }
  }

  // validate the algorithm
  async_square_transpose(matrix_dim, raw_pointer_cast(transposed_matrix.data()), raw_pointer_cast(input_matrix.data())).wait();
  assert(reference_matrix == transposed_matrix);

  // time how long it takes
  double seconds = time_invocation([&]
  {
    async_square_transpose(matrix_dim, raw_pointer_cast(transposed_matrix.data()), raw_pointer_cast(input_matrix.data()));
  });

  // compute bandwidth of the kernel
  double gigabytes = double(2 * matrix_dim * matrix_dim * sizeof(float)) / (1 << 30);
  double bandwidth = gigabytes / seconds;

  std::cout << "Matrix Transpose Bandwidth: " << bandwidth << " GB/s" << std::endl;

  return 0;
}

