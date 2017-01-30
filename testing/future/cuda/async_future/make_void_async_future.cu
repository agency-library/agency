#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>

__managed__ int global_variable;

__global__ void set_global_variable(int value)
{
  global_variable = value;
}

struct return_global_variable
{
  __device__
  int operator()(agency::parallel_agent&) const
  {
    return global_variable;
  }

  __device__
  int operator()() const
  {
    return global_variable;
  }
};

int main()
{
  using namespace agency;

  {
    // test make_async_future() + .wait()
    // from event

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // global_variable starts out at zero
    global_variable = 0;

    // launch a kernel to set the global_variable
    set_global_variable<<<1,1,0,stream>>>(7);

    // record a cuda event
    cudaEventRecord(event, stream);

    // create an async_future to depend on the event
    cuda::async_future<void> future = cuda::experimental::make_async_future(event);
    assert(future.valid());

    // destroy the stream and event
    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    // wait on the future and therefore the kernel launch
    future.wait();

    assert(global_variable == 7);
  }

  {
    // test make_async_future() + .wait()
    // from stream

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // global_variable starts out at zero
    global_variable = 0;

    // launch a kernel to set the global_variable
    set_global_variable<<<1,1,0,stream>>>(7);

    // create an async_future to depend on the stream
    cuda::async_future<void> future = cuda::experimental::make_async_future(stream);
    assert(future.valid());

    // destroy the stream
    cudaStreamDestroy(stream);

    // wait on the future and therefore the kernel launch
    future.wait();

    assert(global_variable == 7);
  }

  {
    // test make_async_future() + .then()
    // from event

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // global_variable starts out at zero
    global_variable = 0;

    // launch a kernel to set the global variable
    set_global_variable<<<1,1,0,stream>>>(7);

    // record a cuda event
    cudaEventRecord(event, stream);

    // create an async_future to depend on the event
    cuda::async_future<void> future = cuda::experimental::make_async_future(event);
    assert(future.valid());

    // destroy the stream and event
    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    // create a continuation depending on the future to return the global variable
    auto result_future = future.then(return_global_variable());

    assert(result_future.get() == 7);
  }

  {
    // test make_async_future() + .then()
    // from stream

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // global_variable starts out at zero
    global_variable = 0;

    // launch a kernel to set the global variable
    set_global_variable<<<1,1,0,stream>>>(7);

    // create an async_future to depend on the event
    cuda::async_future<void> future = cuda::experimental::make_async_future(stream);
    assert(future.valid());

    // destroy the stream
    cudaStreamDestroy(stream);

    // create a continuation depending on the future to return the global variable
    auto result_future = future.then(return_global_variable());

    assert(result_future.get() == 7);
  }

  {
    // test make_async_future() + bulk_then()
    // from event

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // global_variable starts out at zero
    global_variable = 0;

    // launch a kernel to set the global variable
    set_global_variable<<<1,1,0,stream>>>(7);

    // record a cuda event
    cudaEventRecord(event, stream);

    // create an async_future to depend on the event
    cuda::async_future<void> future = cuda::experimental::make_async_future(event);
    assert(future.valid());

    // destroy the stream and event
    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    // create a continuation depending on the future to return the global variable
    auto result_future = bulk_then(cuda::par(1), return_global_variable(), future);

    auto result_container = result_future.get();

    assert(result_container[0] == 7);
  }

  {
    // test make_async_future() + bulk_then()
    // from stream

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // global_variable starts out at zero
    global_variable = 0;

    // launch a kernel to set the global variable
    set_global_variable<<<1,1,0,stream>>>(7);

    // create an async_future to depend on the event
    cuda::async_future<void> future = cuda::experimental::make_async_future(stream);
    assert(future.valid());

    // destroy the stream
    cudaStreamDestroy(stream);

    // create a continuation depending on the future to return the global variable
    auto result_future = bulk_then(cuda::par(1), return_global_variable(), future);

    auto result_container = result_future.get();

    assert(result_container[0] == 7);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

