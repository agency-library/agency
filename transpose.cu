#include <stdio.h>
#include <assert.h>
#include <agency/cuda/execution_policy.hpp>
#include <algorithm>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

// Check errors and print GB/s
void postprocess(const std::vector<float>& ref, const std::vector<float>& res, float ms)
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
__global__ void copy(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

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
    checkCuda(cudaEventCreate(&start_));
    checkCuda(cudaEventCreate(&end_));
    reset();
  }

  void reset()
  {
    checkCuda(cudaEventRecord(start_, stream_));
  }

  float elapsed_milliseconds() const
  {
    checkCuda(cudaEventRecord(end_, stream_));
    checkCuda(cudaEventSynchronize(end_));

    float result = 0;
    checkCuda(cudaEventElapsedTime(&result, start_, end_));
    return result;
  }

  ~cuda_timer()
  {
    checkCuda(cudaEventDestroy(start_));
    checkCuda(cudaEventDestroy(end_));
  }
};


int main(int argc, char **argv)
{
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx*ny*sizeof(float);

  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  checkCuda( cudaSetDevice(devId) );

  std::vector<float> h_idata(nx * ny);
  std::vector<float> h_cdata(nx * ny);
  std::vector<float> h_tdata(nx * ny);
  std::vector<float> gold(nx * ny);
  
  float *d_idata, *d_cdata, *d_tdata;
  checkCuda( cudaMalloc(&d_idata, mem_size) );
  checkCuda( cudaMalloc(&d_cdata, mem_size) );
  checkCuda( cudaMalloc(&d_tdata, mem_size) );

  // check parameters and calculate execution configuration
  if (nx % TILE_DIM || ny % TILE_DIM) {
    throw std::logic_error("nx and ny must be a multiple of TILE_DIM");
  }

  if (TILE_DIM % BLOCK_ROWS) {
    throw std::logic_error("TILE_DIM must be a multiple of BLOCK_ROWS");
  }
    
  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata[j*nx + i] = j*nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j];
  
  // device
  checkCuda( cudaMemcpy(d_idata, h_idata.data(), mem_size, cudaMemcpyHostToDevice) );
  
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
  checkCuda( cudaMemset(d_cdata, 0, mem_size) );
  // warm up
  copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
     copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  }
  ms = timer.elapsed_milliseconds();
  checkCuda( cudaMemcpy(h_cdata.data(), d_cdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(h_idata, h_cdata, ms);

  // -------------
  // copySharedMem 
  // -------------
  printf("%25s", "shared memory copy");
  checkCuda( cudaMemset(d_cdata, 0, mem_size) );
  // warm up
  copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
     copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  }
  ms = timer.elapsed_milliseconds();
  checkCuda( cudaMemcpy(h_cdata.data(), d_cdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(h_idata, h_cdata, ms);

  // --------------
  // transposeNaive 
  // --------------
  printf("%25s", "naive transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_naive{}, d_tdata, d_idata);
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_naive{}, d_tdata, d_idata);
  }
  ms = timer.elapsed_milliseconds();
  checkCuda( cudaMemcpy(h_tdata.data(), d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, ms);

  // ------------------
  // transposeCoalesced 
  // ------------------
  printf("%25s", "coalesced transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_coalesced{}, d_tdata, d_idata);
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_coalesced{}, d_tdata, d_idata);
  }
  ms = timer.elapsed_milliseconds();
  checkCuda( cudaMemcpy(h_tdata.data(), d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, ms);

  // ------------------------
  // transposeNoBankConflicts
  // ------------------------
  printf("%25s", "conflict-free transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_no_bank_conflicts{}, d_tdata, d_idata);
  timer.reset();
  for(int i = 0; i < NUM_REPS; i++)
  {
    agency::cuda::bulk_async(grid({dimGrid.x,dimGrid.y}, {dimBlock.x,dimBlock.y}), transpose_no_bank_conflicts{}, d_tdata, d_idata);
  }
  ms = timer.elapsed_milliseconds();
  checkCuda( cudaMemcpy(h_tdata.data(), d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, ms);
}

