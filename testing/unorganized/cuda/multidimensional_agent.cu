#include <iostream>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>

struct functor
{
  __device__
  void operator()(agency::parallel_agent_2d& self)
  {
    printf("Hello world from agent {%llu, %llu}\n", agency::get<0>(self.index()), agency::get<1>(self.index()));
  }
};

int main()
{
  auto exec = agency::cuda::par2d({0,0}, {2,2});

  agency::bulk_invoke(exec, functor());

  cudaError_t error = cudaDeviceSynchronize();

  std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

  return 0;
}

