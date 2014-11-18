#pragma once

// cf. c++ feature test study group
#if defined(__CUDACC__)
#  if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#    define __cuda_lib_has_cudart 1
#  else
#    define __cuda_lib_has_cudart 0
#  endif
#else
#  define __cuda_lib_has_cudart 0
#endif

#if defined(__CUDACC__)
#  if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 200)
#    define __cuda_lib_has_printf 1
#  else
#    define __cuda_lib_has_printf 0
#  endif
#else
#  define __cuda_lib_has_printf 1
#endif

#include <agency/execution_categories.hpp>
#include <future>
#include <memory>
#include <iostream>
#include <exception>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cstring>
#include <type_traits>
#include "cuda_closure.hpp"


class gpu_id
{
  public:
    typedef int native_handle_type;

    __host__ __device__
    gpu_id(native_handle_type handle)
      : handle_(handle)
    {}

    // default constructor creates a gpu_id which represents no gpu
    __host__ __device__
    gpu_id()
      : gpu_id(-1)
    {}

    // XXX std::this_thread::native_handle() is not const -- why?
    __host__ __device__
    native_handle_type native_handle() const
    {
      return handle_;
    }

    __host__ __device__
    friend inline bool operator==(gpu_id lhs, const gpu_id& rhs)
    {
      return lhs.handle_ == rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator!=(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ != rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator<(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ < rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator<=(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ <= rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator>(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ > rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator>=(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ >= rhs.handle_;
    }

    friend std::ostream& operator<<(std::ostream &os, const gpu_id& id)
    {
      return os << id.native_handle();
    }

  private:
    native_handle_type handle_;
};


__host__ __device__
inline void __terminate()
{
#ifdef __CUDA_ARCH__
  asm("trap;");
#else
  std::terminate();
#endif
}


inline __host__ __device__
void __throw_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
#ifndef __CUDA_ARCH__
    throw thrust::system_error(e, thrust::cuda_category(), message);
#else
#  if (__cuda_lib_has_printf && __cuda_lib_has_cudart)
    printf("Error after %s: %s\n", message, cudaGetErrorString(e));
#  elif __cuda_lib_has_printf
    printf("Error: %s\n", message);
#  endif
    __terminate();
#endif
  }
}


__host__ __device__
gpu_id __this_gpu()
{
  int result = -1;

#if __cuda_lib_has_cudart
  __throw_on_error(cudaGetDevice(&result), "__this_gpu(): cudaGetDevice()");
#endif

  return gpu_id(result);
}


// XXX generalize to multiple arguments
template<typename Arg>
__host__ __device__
cudaError_t __cuda_triple_chevrons(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Arg& arg)
{
#ifndef __CUDA_ARCH__
  cudaConfigureCall(dim3(shape.x), dim3(shape.y), shared_memory_size, stream);
  cudaSetupArgument(arg, 0);
  return cudaLaunch(kernel);
#else
  void *param_buffer = cudaGetParameterBuffer(std::alignment_of<Arg>::value, sizeof(Arg));
  std::memcpy(param_buffer, &arg, sizeof(Arg));
  return cudaLaunchDevice(kernel, param_buffer, dim3(shape.x), dim3(shape.y), shared_memory_size, stream);
#endif // __CUDA_ARCH__
}


template<class Function>
__global__ void __launch_function(Function f)
{
  f(::make_uint2(blockIdx.x, threadIdx.x));
}


// cuda_executor is a BulkExecutor implemented with CUDA kernel launch
class cuda_executor
{
  public:
    using execution_category =
      agency::nested_execution_tag<
        agency::parallel_execution_tag,
        agency::concurrent_execution_tag
      >;


    // XXX shape_type might not be the right name
    //     shape_type is a Tuple-like collection of size_types
    //     the value of each each element specifies the size of a node in the execution hierarchy
    //     the tuple_size<shape_type> must be the same as the nesting depth of execution_category
    //using shape_type = std::uint2;
    using shape_type = ::uint2;


    // this is the type of the parameter handed to functions invoked through bulk_add()
    // XXX threadIdx.x is actually an int
    //     maybe we need to make this int2
    //using index_type = std::uint2;
    using index_type = ::uint2;


    // XXX might want to introduce max_shape (cf. allocator::max_size)
    //     CUDA would definitely take advantage of it


    __host__ __device__
    explicit cuda_executor(int shared_memory_size = 0, cudaStream_t stream = 0, gpu_id gpu = __this_gpu())
      : shared_memory_size_(shared_memory_size),
        stream_(stream),
        gpu_(gpu)
    {}


    __host__ __device__
    int shared_memory_size() const
    {
      return shared_memory_size_;
    }


    __host__ __device__
    cudaStream_t stream() const
    {
      return stream_; 
    }


    __host__ __device__
    gpu_id gpu() const
    {
      return gpu_;
    }


    template<class Function>
    __host__ __device__
    void bulk_add(shape_type shape, Function f, int shared_memory_size, cudaStream_t stream, gpu_id gpu)
    {
      // XXX should refactor all the triple chevron replacement stuff out of this function
      struct workaround
      {
        __host__ __device__
        static void supported_path(void* kernel, shape_type shape, Function f, int shared_memory_size, cudaStream_t stream, gpu_id gpu)
        {
          // reference the kernel to encourage the compiler not to optimize it away
          (void)kernel;

#if __cuda_lib_has_cudart
          if(shape.x > 0 && shape.y > 0)
          {
            // record the current device
            int current_gpu = 0;
            __throw_on_error(cudaGetDevice(&current_gpu), "cuda_executor::bulk_add(): cudaGetDevice()");
            if(current_gpu != gpu.native_handle())
            {
#  ifndef __CUDA_ARCH__
              __throw_on_error(cudaSetDevice(gpu.native_handle()), "cuda_executor::bulk_add(): cudaSetDevice()");
#  else
              __throw_on_error(cudaErrorNotSupported, "cuda_executor::bulk_add(): CUDA kernel launch only allowed on the current device in __device__ code");
#  endif
            }

            // launch the kernel
            __throw_on_error(__cuda_triple_chevrons(kernel, shape, shared_memory_size, stream, f),
#  ifndef __CUDA_ARCH__
              "cuda_executor::bulk_add(): after cudaLaunch()"
#  else
              "cuda_executor::bulk_add(): after cudaLaunchDevice()"
#  endif
            );

            // restore the current device
#  ifndef __CUDA_ARCH__
            if(current_gpu != gpu.native_handle())
            {
              __throw_on_error(cudaSetDevice(current_gpu), "cuda_executor::bulk_add(): cudaSetDevice()");
            }
#  endif // __CUDA_ARCH__
          }
#endif // __cuda_lib_has_cudart
        } // end supported_path

        __host__ __device__
        static void unsupported_path(void* kernel, shape_type shape, Function f, int shared_memory_size, cudaStream_t stream, gpu_id gpu)
        {
          // reference the kernel to encourage the compiler not to optimize it away
          (void)kernel;

          __throw_on_error(cudaErrorNotSupported,
#ifndef __CUDA_ARCH__
            "cuda_executor::bulk_add(): CUDA kernel launch from host requires nvcc"
#else
            "cuda_executor::bulk_add(): CUDA kernel launch from device requires arch=sm_35 or better and rdc=true"
#endif
          );
        } // end unsupported_path
      }; // end workaround

      // unconditionally instantiate the kernel to encourage the compiler not to optimize it away in the unsupported case
      void* kernel = reinterpret_cast<void*>(global_function_pointer<Function>());

#if __cuda_lib_has_cudart
      workaround::supported_path(kernel, shape, f, shared_memory_size, stream, gpu);
#else
      workaround::unsupported_path(kernel, shape, f, shared_memory_size, stream, gpu);
#endif
    }


    template<class Function>
    __host__ __device__
    void bulk_add(shape_type shape, Function f, int shared_memory_size, cudaStream_t stream)
    {
      bulk_add(shape,
               f,
               shared_memory_size,
               stream,
               gpu());
    }


    template<class Function>
    __host__ __device__
    void bulk_add(shape_type shape, Function f, int shared_memory_size)
    {
      bulk_add(shape,
               f,
               shared_memory_size,
               stream());
    }


    template<class Function>
    __host__ __device__
    void bulk_add(shape_type shape, Function f)
    {
      bulk_add(shape, f, shared_memory_size(), stream());
    }


    // this is exposed because it's necessary if a client wants to compute occupancy
    // alternatively, cuda_executor could report occupancy of a Function for a given block size
    template<class Function>
    __host__ __device__
    static decltype(&__launch_function<Function>) global_function_pointer()
    {
      return &__launch_function<Function>;
    }


  private:
    int shared_memory_size_;
    cudaStream_t stream_;
    gpu_id gpu_;
};


void __notify(cudaStream_t stream, cudaError_t status, void* data)
{
  std::unique_ptr<std::promise<void>> promise(reinterpret_cast<std::promise<void>*>(data));

  promise->set_value();
}


// XXX can't make this __host__ __device__ due to the std::future
template<class Function, class... Args>
std::future<void> bulk_async(cuda_executor& ex, typename cuda_executor::shape_type shape, Function&& f, Args&&... args)
{
  // add to the executor
  ex.bulk_add(shape, make_cuda_closure(std::forward<Function>(f), std::forward<Args>(args)...));

  // XXX unique_ptr & promise won't be valid in __device__ code
  std::unique_ptr<std::promise<void>> promise(new std::promise<void>());

  auto result = promise->get_future();

  // call __notify when kernel is finished
  // XXX add error checking
  // XXX cudaStreamAddCallback probably isn't valid in __device__ code
  __throw_on_error(cudaStreamAddCallback(ex.stream(), __notify, promise.release(), 0),
                   "bulk_async(): cudaStreamAddcallback");

  return result;
}


// XXX could probably make this __host__ __device__
template<class Function, class... Args>
void bulk_invoke(cuda_executor& ex, typename cuda_executor::shape_type shape, Function&& f, Args&&... args)
{
  // XXX might be a more efficient way to do this by simply synchronizing with ex.stream()
  bulk_async(ex, shape, std::forward<Function>(f), std::forward<Args>(args)...).wait();
}

