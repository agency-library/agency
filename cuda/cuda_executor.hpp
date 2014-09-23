#pragma once

#include <execution_categories>
#include <future>
#include <memory>
#include <iostream>
#include <exception>
#include <cstring>
#include <type_traits>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/tuple.h>
#include "feature_test.hpp"
#include "bind.hpp"
#include "unique_cuda_ptr.hpp"
#include "terminate.hpp"


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


inline void __cuda_setup_arguments(size_t){}

template<class Arg1, class... Args>
void __cuda_setup_arguments(size_t offset, const Arg1& arg1, const Args&... args)
{
  cudaSetupArgument(arg1, offset);
  __cuda_setup_arguments(offset + sizeof(Arg1), args...);
}


template<class T1, class... Ts>
struct __first_type
{
  using type = T1;
};


template<class Arg1, class... Args>
__host__ __device__
auto __first_parameter(Arg1&& arg1, Args&&... args)
  -> decltype(std::forward<Arg1>(arg1))
{
  return std::forward<Arg1>(arg1);
}


template<typename... Args>
__host__ __device__
cudaError_t __cuda_triple_chevrons(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
{
  // reference the kernel to encourage the compiler not to optimize it away
  (void)kernel;

#if __cuda_lib_has_cudart
#  ifndef __CUDA_ARCH__
  cudaConfigureCall(dim3(shape.x), dim3(shape.y), shared_memory_size, stream);
  __cuda_setup_arguments(0, args...);
  return cudaLaunch(kernel);
#  else
  // XXX generalize to multiple arguments
  if(sizeof...(Args) != 1)
  {
    return cudaErrorNotSupported;
  }

  using Arg = typename __first_type<Args...>::type;

  void *param_buffer = cudaGetParameterBuffer(std::alignment_of<Arg>::value, sizeof(Arg));
  std::memcpy(param_buffer, &__first_parameter(args...), sizeof(Arg));
  return cudaLaunchDevice(kernel, param_buffer, dim3(shape.x), dim3(shape.y), shared_memory_size, stream);
#  endif // __CUDA_ARCH__
#else // __cuda_lib_has_cudart
  return cudaErrorNotSupported;
#endif
}


template<class... Args>
__host__ __device__
cudaError_t __launch_cuda_kernel(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
{
  struct workaround
  {
    __host__ __device__
    static cudaError_t supported_path(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
    {
      // reference the kernel to encourage the compiler not to optimize it away
      (void)kernel;

      return __cuda_triple_chevrons(kernel, shape, shared_memory_size, stream, args...);
    }

    __host__ __device__
    static cudaError_t unsupported_path(void* kernel, uint2, int, cudaStream_t, const Args&...)
    {
      // reference the kernel to encourage the compiler not to optimize it away
      (void)kernel;

      return cudaErrorNotSupported;
    }
  };

#if __cuda_lib_has_cudart
  cudaError_t result = workaround::supported_path(kernel, shape, shared_memory_size, stream, args...);
#else
  cudaError_t result = workaround::unsupported_path(kernel, shape, shared_memory_size, stream, args...);
#endif

  return result;
}


template<class... Args>
__host__ __device__
void __checked_launch_cuda_kernel(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, const Args&... args)
{
  // the error message we return depends on how the program was compiled
  const char* error_message = 
#if __cuda_lib_has_cudart
   // we have access to CUDART, so something went wrong during the kernel
#  ifndef __CUDA_ARCH__
   "__checked_launch_cuda_kernel(): CUDA error after cudaLaunch()"
#  else
   "__checked_launch_cuda_kernel(): CUDA error after cudaLaunchDevice()"
#  endif // __CUDA_ARCH__
#else // __cuda_lib_has_cudart
   // we don't have access to CUDART, so output a useful error message explaining why it's unsupported
#  ifndef __CUDA_ARCH__
   "__checked_launch_cuda_kernel(): CUDA kernel launch from host requires nvcc"
#  else
   "__checked_launch_cuda_kernel(): CUDA kernel launch from device requires arch=sm_35 or better and rdc=true"
#  endif // __CUDA_ARCH__
#endif
  ;

  __throw_on_error(__launch_cuda_kernel(kernel, shape, shared_memory_size, stream, args...), error_message);
}


template<class... Args>
__host__ __device__
void __checked_launch_cuda_kernel_on_device(void* kernel, uint2 shape, int shared_memory_size, cudaStream_t stream, int device, const Args&... args)
{
#if __cuda_lib_has_cudart
  // record the current device
  int current_device = 0;
  __throw_on_error(cudaGetDevice(&current_device), "__checked_launch_cuda_kernel_on_device(): cudaGetDevice()");
  if(current_device != device)
  {
#  ifndef __CUDA_ARCH__
    __throw_on_error(cudaSetDevice(device), "__checked_launch_cuda_kernel_on_device(): cudaSetDevice()");
#  else
    __throw_on_error(cudaErrorNotSupported, "__checked_launch_cuda_kernel_on_device(): CUDA kernel launch only allowed on the current device in __device__ code");
#  endif // __CUDA_ARCH__
  }
#else
  // the error message we return depends on how the program was compiled
  const char* error_message = 
#  ifndef __CUDA_ARCH__
     "__checked_launch_cuda_kernel_on_device(): CUDA kernel launch from host requires nvcc"
#  else
     "__checked_launch_cuda_kernel_on_device(): CUDA kernel launch from device requires arch=sm_35 or better and rdc=true"
#  endif
  ;
  __throw_on_error(cudaErrorNotSupported, error_message);
#endif // __cuda_lib_has_cudart

  __checked_launch_cuda_kernel(kernel, shape, shared_memory_size, stream, args...);

#if __cuda_lib_has_cudart
  // restore the device
#  ifndef __CUDA_ARCH__
  if(current_device != device)
  {
    __throw_on_error(cudaSetDevice(current_device), "__checked_launch_cuda_kernel_on_device: cudaSetDevice()");
  }
#  endif // __CUDA_ARCH__
#else
  __throw_on_error(cudaErrorNotSupported, "__checked_launch_cuda_kernel_on_device(): cudaSetDevice requires CUDART");
#endif // __cuda_lib_has_cudart
}


template<class Function>
__global__ void __launch_function(Function f)
{
  uint2 idx = make_uint2(blockIdx.x, threadIdx.x);
  f(idx);
}


void __notify(cudaStream_t stream, cudaError_t status, void* data)
{
  std::unique_ptr<std::promise<void>> promise(reinterpret_cast<std::promise<void>*>(data));

  promise->set_value();
}


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
gpu_id __this_gpu()
{
  int result = -1;

#if __cuda_lib_has_cudart
  __throw_on_error(cudaGetDevice(&result), "__this_gpu(): cudaGetDevice()");
#endif

  return gpu_id(result);
}


// give CUDA built-in vector types Tuple-like access
template<std::size_t i>
__host__ __device__
unsigned int&
  get(uint2& x)
{
  return reinterpret_cast<unsigned int*>(&x)[i];
}


template<std::size_t i>
__host__ __device__
const unsigned int&
  get(const uint2& x)
{
  return reinterpret_cast<const unsigned int*>(&x)[i];
}


namespace std
{


template<>
struct tuple_size<uint2> : integral_constant<size_t,2> {};


template<size_t i>
struct tuple_element<i,uint2>
{
  using type = unsigned int;
};


} // end std


template<class Function, class OuterSharedType, class InnerSharedType>
struct __function_with_shared_arguments
{
  __host__ __device__
  __function_with_shared_arguments(Function f, OuterSharedType* outer_ptr, InnerSharedType inner_shared_init)
    : f_(f),
      outer_ptr_(outer_ptr),
      inner_shared_init_(inner_shared_init)
  {}

  template<class Agent>
  __device__
  void operator()(Agent& agent)
  {
    __shared__ InnerSharedType inner_param;

    // initialize the inner shared parameter
    if(agent.y == 0)
    {
      printf("initializing inner parameter with %d\n", inner_shared_init_);
      inner_param = inner_shared_init_;
    }
    __syncthreads();

    thrust::tuple<OuterSharedType&,InnerSharedType&> shared_params(*outer_ptr_, inner_param);

    f_(agent, shared_params);
  }

  Function         f_;
  OuterSharedType* outer_ptr_;
  InnerSharedType  inner_shared_init_;
};


template<class OuterSharedType>
struct __copy_outer_shared_parameter
{
  __host__ __device__
  __copy_outer_shared_parameter(OuterSharedType* outer_shared_ptr, const OuterSharedType& outer_shared_init)
    : outer_shared_ptr_(outer_shared_ptr),
      outer_shared_init_(outer_shared_init)
  {}

  __device__
  void operator()(uint2)
  {
    new (outer_shared_ptr_) OuterSharedType(outer_shared_init_);
  }

  OuterSharedType* outer_shared_ptr_;
  OuterSharedType  outer_shared_init_;
};


// cuda_executor is a BulkExecutor implemented with CUDA kernel launch
class cuda_executor
{
  public:
    using execution_category =
      std::nested_execution_tag<
        std::parallel_execution_tag,
        std::concurrent_execution_tag
      >;


    // XXX shape_type might not be the right name
    //     shape_type is a Tuple-like collection of size_types
    //     the value of each each element specifies the size of a node in the execution hierarchy
    //     the tuple_size<shape_type> must be the same as the nesting depth of execution_category
    //using shape_type = std::uint2;
    using shape_type = uint2;


    // this is the type of the parameter handed to functions invoked through bulk_add()
    // XXX threadIdx.x is actually an int
    //     maybe we need to make this int2
    //using index_type = std::uint2;
    using index_type = uint2;


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
    std::future<void> bulk_async(Function f, shape_type shape)
    {
      launch(f, shape);

      void* kernel = reinterpret_cast<void*>(global_function_pointer<Function>());

      // XXX unique_ptr & promise won't be valid in __device__ code
      std::unique_ptr<std::promise<void>> promise(new std::promise<void>());
    
      auto result = promise->get_future();
    
      // call __notify when kernel is finished
      // XXX cudaStreamAddCallback probably isn't valid in __device__ code
      __throw_on_error(cudaStreamAddCallback(stream(), __notify, promise.release(), 0),
                       "cuda_executor::bulk_async(): cudaStreamAddCallback");
    
      return result;
    }


    template<class Function, class Tuple>
    std::future<void> bulk_async(Function f, shape_type shape, Tuple shared_arg_tuple)
    {
      using outer_shared_type = typename thrust::tuple_element<0,Tuple>::type;
      using inner_shared_type = typename thrust::tuple_element<1,Tuple>::type;

      // XXX wrap all this up into make_unique_cuda
      // allocate outer shared argument
      unique_cuda_ptr<outer_shared_type> outer_shared_arg(thrust::cuda::malloc<outer_shared_type>(1));

      // copy construct the outer shared arg
      // XXX do this asynchronously
      //     don't do this if outer_shared_type is std::ignore
      bulk_invoke(__copy_outer_shared_parameter<outer_shared_type>(outer_shared_arg.get(), get<0>(shared_arg_tuple)), shape_type(1,1));

      // wrap up f in a thing that will marshal the shared arguments to it
      // note the .release()
      auto g = __function_with_shared_arguments<Function, outer_shared_type, inner_shared_type>(f, outer_shared_arg.release(), get<1>(shared_arg_tuple));

      // XXX to deallocate & destroy the outer_shared_arg, we need to do a bulk_async(...).then(...)
      //     for now it just leaks :(

      return bulk_async(g, shape);
    }


    template<class Function>
    __host__ __device__
    void bulk_invoke(Function f, shape_type shape)
    {
#ifndef __CUDA_ARCH__
      bulk_async(f, shape).wait();
#else
      launch(f, shape);

#  if __cuda_lib_has_cudart
      __throw_on_error(cudaDeviceSynchronize(), "cuda_executor::bulk_invoke(): cudaDeviceSynchronize");
#  endif
#endif
    }


    template<class Function, class Tuple>
    __host__ __device__
    void bulk_invoke(Function f, shape_type shape, Tuple shared_arg_tuple)
    {
      using outer_shared_type = typename thrust::tuple_element<0,Tuple>::type;
      using inner_shared_type = typename thrust::tuple_element<1,Tuple>::type;

      // allocate outer shared argument
      unique_cuda_ptr<outer_shared_type> outer_shared_arg(thrust::cuda::malloc<outer_shared_type>(1));

      // copy construct the outer shared arg
      // XXX don't do this if outer_shared_type is std::ignore
      bulk_invoke(__copy_outer_shared_parameter<outer_shared_type>(outer_shared_arg.get(), get<0>(shared_arg_tuple)), shape_type{1,1});

      // wrap up f in a thing that will marshal the shared arguments to it
      auto g = __function_with_shared_arguments<Function, outer_shared_type, inner_shared_type>(f, outer_shared_arg.get(), get<1>(shared_arg_tuple));

      return bulk_invoke(g, shape);
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
    template<class Function>
    __host__ __device__
    void launch(Function f, shape_type shape)
    {
      launch(f, shape, shared_memory_size());
    }

    template<class Function>
    __host__ __device__
    void launch(Function f, shape_type shape, int shared_memory_size)
    {
      launch(f, shape, shared_memory_size, stream());
    }

    template<class Function>
    __host__ __device__
    void launch(Function f, shape_type shape, int shared_memory_size, cudaStream_t stream)
    {
      launch(f, shape, shared_memory_size, stream, gpu());
    }

    template<class Function>
    __host__ __device__
    void launch(Function f, shape_type shape, int shared_memory_size, cudaStream_t stream, gpu_id gpu)
    {
      void* kernel = reinterpret_cast<void*>(global_function_pointer<Function>());

      __checked_launch_cuda_kernel_on_device(kernel, shape, shared_memory_size, stream, gpu.native_handle(), f);
    }

    int shared_memory_size_;
    cudaStream_t stream_;
    gpu_id gpu_;
};


// XXX could probably make this __host__ __device__
template<class Function, class... Args>
__host__ __device__
void bulk_invoke(cuda_executor& ex, typename cuda_executor::shape_type shape, Function&& f, Args&&... args)
{
  auto g = thrust::experimental::bind(std::forward<Function>(f), thrust::placeholders::_1, std::forward<Args>(args)...);
  ex.bulk_invoke(g, shape);
}

