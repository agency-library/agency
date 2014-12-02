#include <iostream>
#include <agency/execution_policy.hpp>
#include <agency/coordinate.hpp>
#include "cuda/execution_policy.hpp"

template<class ExecutionCategory, size_t N>
using basic_multidimensional_agent = agency::detail::basic_execution_agent<ExecutionCategory, agency::point<size_t,N>>;

using parallel_agent_2d = basic_multidimensional_agent<agency::parallel_execution_tag, 2>;

const cuda::detail::basic_execution_policy<parallel_agent_2d, cuda::parallel_executor> par2d{};

struct functor
{
  __device__
  void operator()(parallel_agent_2d& self)
  {
    printf("Hello world from agent {%d, %d}\n", std::get<0>(self.index()), std::get<1>(self.index()));
  }
};

int main()
{
  auto exec = par2d({0,0}, {2,2});

  cuda::bulk_invoke(exec, functor());

  cudaError_t error = cudaDeviceSynchronize();

  std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

  return 0;
}

