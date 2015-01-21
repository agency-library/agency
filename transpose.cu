#include <cstdio>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <agency/cuda/execution_policy.hpp>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

// Check errors and print GB/s
void postprocess(const thrust::host_vector<float>& ref, const thrust::host_vector<float>& res, float ms)
{
  auto mismatch = std::mismatch(ref.begin(), ref.end(), res.begin());
  if(mismatch.first != ref.end())
  {
    int i = mismatch.first - ref.begin();
    printf("%d %f %f\n", i, *mismatch.second, *mismatch.first);
    printf("%25s\n", "*** FAILED ***");
  }
  else
  {
    printf("%20.2f\n", 2 * ref.size() * sizeof(float) * 1e-6 * NUM_REPS / ms );
  }
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
struct copy_kernel
{
  template<class Agent>
  __device__
  void operator()(Agent& self, float* odata, const float* idata)
  {
    auto idx = TILE_DIM * self.outer().index() + self.inner().index();
    int width = self.outer().group_shape()[0] * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    {
      odata[(idx[1]+j)*width + idx[0]] = idata[(idx[1]+j)*width + idx[0]];
    }
  }
};


// copy kernel using shared memory
// Also used as reference case, demonstrating effect of using shared memory.
__global__ void copySharedMem(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
  
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
}


struct transpose_naive
{
  template<class Agent>
  __device__
  void operator()(Agent& self, float* odata, const float* idata)
  {
    auto idx = TILE_DIM * self.outer().index() + self.inner().index();
    int width = self.outer().group_shape()[0] * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      odata[idx[0]*width + (idx[1]+j)] = idata[(idx[1]+j)*width + idx[0]];
  }
};


struct transpose_coalesced
{
  template<class Agent>
  __device__
  void operator()(Agent& self, float* odata, const float* idata)
  {
    __shared__ float tile[TILE_DIM][TILE_DIM];
      
    int x = self.outer().index()[0] * TILE_DIM + self.inner().index()[0];
    int y = self.outer().index()[1] * TILE_DIM + self.inner().index()[1];
    int width = self.outer().group_shape()[0] * TILE_DIM;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      tile[self.inner().index()[1]+j][self.inner().index()[0]] = idata[(y+j)*width + x];
    }

    self.inner().wait();

    x = self.outer().index()[1] * TILE_DIM + self.inner().index()[0];  // transpose block offset
    y = self.outer().index()[0] * TILE_DIM + self.inner().index()[1];

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      odata[(y+j)*width + x] = tile[self.inner().index()[0]][self.inner().index()[1] + j];
    }
  }
};


struct transpose_no_bank_conflicts
{
  template<class Agent>
  __device__
  void operator()(Agent& self, float* odata, const float* idata)
  {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
      
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
       tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    self.inner().wait();

    x = self.outer().index()[1] * TILE_DIM + self.inner().index()[0];  // transpose block offset
    y = self.outer().index()[0] * TILE_DIM + self.inner().index()[1];

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
       odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
};


// XXX need to figure out how to make this par(con) select grid_executor_2d automatically
auto grid(agency::size2 num_blocks, agency::size2 num_threads)
  -> decltype(agency::cuda::par(num_blocks, agency::cuda::con(num_threads)).on(agency::cuda::grid_executor_2d{}))
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads)).on(agency::cuda::grid_executor_2d{});
}


struct cuda_timer
{
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t end_;

  cuda_timer(cudaStream_t stream = 0) : stream_(stream), start_{0}, end_{0}
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
    reset();
  }

  void reset()
  {
    cudaEventRecord(start_, stream_);
  }

  float elapsed_milliseconds() const
  {
    cudaEventRecord(end_, stream_);
    cudaEventSynchronize(end_);

    float result = 0;
    cudaEventElapsedTime(&result, start_, end_);
    return result;
  }

  ~cuda_timer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(end_);
  }
};


int main(int argc, char **argv)
{
  const int nx = 1024;
  const int ny = 1024;

  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devId);
  printf("\nDevice : %s\n", prop.name);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  cudaSetDevice(devId);

  thrust::host_vector<float> h_idata(nx * ny);
  thrust::host_vector<float> h_cdata(nx * ny);
  thrust::host_vector<float> h_tdata(nx * ny);
  thrust::host_vector<float> gold(nx * ny);

  thrust::device_vector<float> d_idata(nx * ny);
  thrust::device_vector<float> d_cdata(nx * ny);
  thrust::device_vector<float> d_tdata(nx * ny);
  
  // check parameters and calculate execution configuration
  if (nx % TILE_DIM || ny % TILE_DIM) {
    throw std::logic_error("nx and ny must be a multiple of TILE_DIM");
  }

  if (TILE_DIM % BLOCK_ROWS) {
    throw std::logic_error("TILE_DIM must be a multiple of BLOCK_ROWS");
  }
    
  // initialize input
  std::iota(h_idata.begin(), h_idata.end(), 0);

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j];
  
  // copy input to device
  d_idata = h_idata;
  
  float ms;
  cuda_timer timer;

  // ------------
  // time kernels
  // ------------
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
  
  // ----
  // copy 
  // ----
  printf("%25s", "copy");
  thrust::fill(d_cdata.begin(), d_cdata.end(), 0);
  // warm up
  agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), copy_kernel{}, raw_pointer_cast(d_cdata.data()), raw_pointer_cast(d_idata.data()));
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), copy_kernel{}, raw_pointer_cast(d_cdata.data()), raw_pointer_cast(d_idata.data()));
  }
  ms = timer.elapsed_milliseconds();
  h_cdata = d_cdata;
  postprocess(h_idata, h_cdata, ms);

  // -------------
  // copySharedMem 
  // -------------
  printf("%25s", "shared memory copy");
  thrust::fill(d_cdata.begin(), d_cdata.end(), 0);
  // warm up
  copySharedMem<<<dimGrid, dimBlock>>>(raw_pointer_cast(d_cdata.data()), raw_pointer_cast(d_idata.data()));
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    copySharedMem<<<dimGrid, dimBlock>>>(raw_pointer_cast(d_cdata.data()), raw_pointer_cast(d_idata.data()));
  }
  ms = timer.elapsed_milliseconds();
  h_cdata = d_cdata;
  postprocess(h_idata, h_cdata, ms);

  // --------------
  // transposeNaive 
  // --------------
  printf("%25s", "naive transpose");
  thrust::fill(d_tdata.begin(), d_tdata.end(), 0);
  // warmup
  agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_naive{}, raw_pointer_cast(d_tdata.data()), raw_pointer_cast(d_idata.data()));
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_naive{}, raw_pointer_cast(d_tdata.data()), raw_pointer_cast(d_idata.data()));
  }
  ms = timer.elapsed_milliseconds();
  h_tdata = d_tdata;
  postprocess(gold, h_tdata, ms);

  // ------------------
  // transposeCoalesced 
  // ------------------
  printf("%25s", "coalesced transpose");
  thrust::fill(d_tdata.begin(), d_tdata.end(), 0);
  // warmup
  agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_coalesced{}, raw_pointer_cast(d_tdata.data()), raw_pointer_cast(d_idata.data()));
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_coalesced{}, raw_pointer_cast(d_tdata.data()), raw_pointer_cast(d_idata.data()));
  }
  ms = timer.elapsed_milliseconds();
  h_tdata = d_tdata;
  postprocess(gold, h_tdata, ms);

  // ------------------------
  // transposeNoBankConflicts
  // ------------------------
  printf("%25s", "conflict-free transpose");
  thrust::fill(d_tdata.begin(), d_tdata.end(), 0);
  // warmup
  agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_no_bank_conflicts{}, raw_pointer_cast(d_tdata.data()), raw_pointer_cast(d_idata.data()));
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_no_bank_conflicts{}, raw_pointer_cast(d_tdata.data()), raw_pointer_cast(d_idata.data()));
  }
  ms = timer.elapsed_milliseconds();
  h_tdata = d_tdata;
  postprocess(gold, h_tdata, ms);

  return 0;
}

