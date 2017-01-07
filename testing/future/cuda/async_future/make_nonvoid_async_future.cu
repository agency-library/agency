#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>


__global__ void set_value(int* ptr_to_value, int to)
{
  *ptr_to_value = to;
}


struct add_thirteen
{
  __device__
  int operator()(agency::parallel_agent&, int& predecessor) const
  {
    return predecessor + 13;
  }

  __device__
  int operator()(int& predecessor) const
  {
    return predecessor + 13;
  }
};


std::atomic<int> num_deleter_calls{0};


template<class T>
struct my_deleter
{
  agency::cuda::allocator<int> alloc;

  __AGENCY_ANNOTATION
  void operator()(int *ptr)
  {
    alloc.deallocate(ptr, 1);

#ifndef __CUDA_ARCH__
    // XXX it's difficult to validate that this deleter actually gets called because
    //     it can be called after main() ends
    //     moreover, the program can simply end before it's called at all
    //     instead of attempting to count the number of calls made to the deleter,
    //     just print to the terminal
    std::cout << "my_deleter::operator()" << std::endl;
#endif
  }
};


int main()
{
  using namespace agency;

  {
    // test make_async_future() + .get()

    // allocate the non-void value
    agency::cuda::allocator<int> alloc;
    int* ptr_to_value = alloc.allocate(1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // launch a kernel to set the value
    set_value<<<1,1,0,stream>>>(ptr_to_value, 7);

    // record a cuda event
    cudaEventRecord(event, stream);

    // create an async_future to depend on the event
    cuda::async_future<int> future = cuda::experimental::make_async_future(event, ptr_to_value, my_deleter<int>{alloc});
    assert(future.valid());

    // destroy the stream and event
    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    assert(future.get() == 7);
  }

  {
    // test make_async_future() + .then()

    // allocate the non-void value
    agency::cuda::allocator<int> alloc;
    int* ptr_to_value = alloc.allocate(1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // launch a kernel to set the global variable
    set_value<<<1,1,0,stream>>>(ptr_to_value, 7);

    // record a cuda event
    cudaEventRecord(event, stream);

    // create an async_future to depend on the event
    cuda::async_future<int> future = cuda::experimental::make_async_future(event, ptr_to_value, my_deleter<int>{alloc});
    assert(future.valid());

    // destroy the stream and event
    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    // create a continuation depending on the future to return the global variable
    auto result_future = future.then(add_thirteen());

    assert(result_future.get() == 7 + 13);
  }

  {
    // test make_async_future() + bulk_then()

    // allocate the non-void value
    agency::cuda::allocator<int> alloc;
    int* ptr_to_value = alloc.allocate(1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // launch a kernel to set the global variable
    set_value<<<1,1,0,stream>>>(ptr_to_value, 7);

    // record a cuda event
    cudaEventRecord(event, stream);

    // create an async_future to depend on the event
    cuda::async_future<int> future = cuda::experimental::make_async_future(event, ptr_to_value, my_deleter<int>{alloc});
    assert(future.valid());

    // destroy the stream and event
    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    // create a continuation depending on the future to return the global variable
    auto result_future = bulk_then(cuda::par(1), add_thirteen(), future);

    auto result_container = result_future.get();

    assert(result_container[0] == 7 + 13);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

