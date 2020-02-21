// This example program demonstrates how to use CUDA extended device lambdas with Agency.
#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <utility>
#include <cstdio>

// This function demonstrates how to use a CUDA extended device lambda
// which does not return a result.
void use_device_lambda_with_no_result()
{
  // because extended device lambda is a CUDA feature, we need to use
  // an execution policy in agency::cuda
  auto policy = agency::cuda::par(2);

  // when a device lambda returns no result, we can use them directly
  // with functions like agency::bulk_invoke()
  std::cout << "Invoking device lambda on the GPU" << std::endl;
  agency::bulk_invoke(policy, [] __device__ (agency::parallel_agent& self)
  {
    printf("agent %llu: Hello, world from device lambda!\n", self.index());
  });
}


// This wrapper functor helps us adapt result-returning device lambdas
// for use with Agency.
template<class Result, class Function>
struct wrapper
{
  Function f;

  template<class... Args>
  __host__ __device__
  Result operator()(Args&&... args)
  {
    return f(std::forward<Args>(args)...);
  }
};

template<class Result, class Function>
wrapper<Result,Function> wrap(Function f)
{
  return wrapper<Result,Function>{f};
}


// this function demonstrates how to use a CUDA extended device lambda
// which does return a result.
void use_device_lambda_with_result()
{
  auto policy = agency::cuda::par(2);

  // CUDA device lambdas have limitations as discussed in the CUDA C++ Programming Guide
  // when a device lambda returns a result, we need to adapt them
  // in order to use them with functions like agency::bulk_invoke()
  auto device_lambda_with_result = [] __device__ (agency::parallel_agent& self)
  {
    printf("agent %llu: Hello, world! This lambda must wrapped for Agency to return its results.\n", self.index());

    return self.index();
  };

  // functions like agency::bulk_invoke() will ignore the return type of CUDA device
  // when they are passed directly

  // to adapt a device lambda which returns a result to work with Agency,
  // we need to wrap it in some other functor. Note how we explicitly
  // name the type of result returned by the lambda as a template parameter
  // to our wrap<T>() function
  auto wrapped_device_lambda = wrap<int>(device_lambda_with_result);

  // now that we've adapted the integer-returning device lambda we can use it with
  // functions like agency::bulk_invoke() and receive their results
  std::cout << "Invoking wrapped device lambda on the GPU" << std::endl;
  auto results = agency::bulk_invoke(policy, wrapped_device_lambda);

  // XXX disabled due to nvbug 1759492
  //std::vector<int> expected_results{0, 1};
  //assert(results == expected_results);
}


// This function demonstrates how to use a CUDA extended host device lambda with Agency.
void use_host_device_lambda()
{
  // CUDA host device lambdas do not share device lambdas' limitations
  // they interoperate with Agency normally

  auto host_device_lambda = [] __host__ __device__ (agency::parallel_agent& self)
  {
    printf("agent %zu: Hello, world from host device lambda!\n", self.index());
    return self.index();
  };

  // XXX disabled due to nvbug 1759492
  //std::vector<int> expected_results{0,1};

  // we can invoke host device lambdas using CUDA execution policies
  std::cout << "Invoking host device lambda on the GPU" << std::endl;
  auto gpu_results = agency::bulk_invoke(agency::cuda::par(2), host_device_lambda);

  // XXX disabled due to nvbug 1759492
  //assert(gpu_results == expected_results);

  std::cout << std::endl;

  // or we can invoke host device lambdas using non-CUDA execution policies
  std::cout << "Invoking host device lambda on the CPU" << std::endl;
  auto cpu_results = agency::bulk_invoke(agency::par(2), host_device_lambda);

  // XXX disabled due to nvbug 1759492
  //assert(cpu_results == expected_results);
}


int main()
{
  use_device_lambda_with_no_result();

  std::cout << std::endl;

  use_device_lambda_with_result();

  std::cout << std::endl;

  use_host_device_lambda();

  return 0;
}

