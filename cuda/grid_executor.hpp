#pragma once

#include <execution_categories>
#include <future>
#include <memory>
#include <iostream>
#include <exception>
#include <cstring>
#include <type_traits>
#include <cassert>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <flattened_executor>
#include <thrust/detail/minmax.h>
#include "thrust_tuple_cpp11.hpp"
#include "feature_test.hpp"
#include "gpu.hpp"
#include "bind.hpp"
#include "unique_cuda_ptr.hpp"
#include "terminate.hpp"
#include "uninitialized.hpp"
#include "detail/launch_kernel.hpp"


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


namespace cuda
{
namespace detail
{


template<class Function, class OuterSharedType, class InnerSharedType>
struct function_with_shared_arguments
{
  __host__ __device__
  function_with_shared_arguments(Function f, OuterSharedType* outer_ptr, InnerSharedType inner_shared_init)
    : f_(f),
      outer_ptr_(outer_ptr),
      inner_shared_init_(inner_shared_init)
  {}

  template<class Agent>
  __device__
  void operator()(Agent& agent)
  {
    // XXX can't rely on a default constructor
    __shared__ cuda::uninitialized<InnerSharedType> inner_param;

    // initialize the inner shared parameter
    if(agent.y == 0)
    {
      inner_param.construct(inner_shared_init_);
    }
    __syncthreads();

    thrust::tuple<OuterSharedType&,InnerSharedType&> shared_params(*outer_ptr_, inner_param);

    f_(agent, shared_params);

    __syncthreads();

    // destroy the inner shared parameter
    if(agent.y == 0)
    {
      inner_param.destroy();
    }
  }

  Function         f_;
  OuterSharedType* outer_ptr_;
  InnerSharedType  inner_shared_init_;
};


template<class OuterSharedType>
struct copy_outer_shared_parameter
{
  __host__ __device__
  copy_outer_shared_parameter(OuterSharedType* outer_shared_ptr, const OuterSharedType& outer_shared_init)
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


template<class Function>
__global__ void grid_executor_kernel(Function f)
{
  uint2 idx = make_uint2(blockIdx.x, threadIdx.x);
  f(idx);
}


void grid_executor_notify(cudaStream_t stream, cudaError_t status, void* data)
{
  std::unique_ptr<std::promise<void>> promise(reinterpret_cast<std::promise<void>*>(data));

  promise->set_value();
}


} // end detail


// grid_executor is a BulkExecutor implemented with CUDA kernel launch
class grid_executor
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
    // XXX for cuda, maybe this should be int2?
    using shape_type = uint2;


    // this is the type of the parameter handed to functions invoked through bulk_add()
    // XXX threadIdx.x is actually an int
    //     maybe we need to make this int2
    //using index_type = std::uint2;
    using index_type = uint2;

    template<class Tuple>
    using shared_param_type = __thrust_tuple_of_references_t<Tuple>;


    __host__ __device__
    explicit grid_executor(int shared_memory_size = 0, cudaStream_t stream = 0, gpu_id gpu = detail::current_gpu())
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
    shape_type max_shape(Function) const
    {
      shape_type result = {0,0};

      auto fun_ptr = global_function_pointer<Function>();
      (void)fun_ptr;

#if __cuda_lib_has_cudart
      // record the current device
      int current_device = 0;
      __throw_on_error(cudaGetDevice(&current_device), "cuda::grid_executor::max_shape(): cudaGetDevice()");
      if(current_device != gpu().native_handle())
      {
#  ifndef __CUDA_ARCH__
        __throw_on_error(cudaSetDevice(gpu().native_handle()), "cuda::grid_executor::max_shape(): cudaSetDevice()");
#  else
        __throw_on_error(cudaErrorNotSupported, "cuda::grid_executor::max_shape(): cudaSetDevice only allowed in __host__ code");
#  endif // __CUDA_ARCH__
      }

      int max_block_dimension_x = 0;
      __throw_on_error(cudaDeviceGetAttribute(&max_block_dimension_x, cudaDevAttrMaxBlockDimX, gpu().native_handle()),
                       "cuda::grid_executor::max_shape(): cudaDeviceGetAttribute");

      cudaFuncAttributes attr{};
      __throw_on_error(cudaFuncGetAttributes(&attr, fun_ptr),
                       "cuda::grid_executor::max_shape(): cudaFuncGetAttributes");

      // restore current device
      if(current_device != gpu().native_handle())
      {
#  ifndef __CUDA_ARCH__
        __throw_on_error(cudaSetDevice(current_device), "cuda::grid_executor::max_shape(): cudaSetDevice()");
#  else
        __throw_on_error(cudaErrorNotSupported, "cuda::grid_executor::max_shape(): cudaSetDevice only allowed in __host__ code");
#  endif // __CUDA_ARCH__
      }

      result = shape_type{static_cast<unsigned int>(max_block_dimension_x), static_cast<unsigned int>(attr.maxThreadsPerBlock)};
#endif // __cuda_lib_has_cudart

      return result;
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
      __throw_on_error(cudaStreamAddCallback(stream(), detail::grid_executor_notify, promise.release(), 0),
                       "cuda::grid_executor::bulk_async(): cudaStreamAddCallback");
    
      return result;
    }


    template<class Function, class Tuple>
    std::future<void> bulk_async(Function f, shape_type shape, Tuple shared_arg_tuple)
    {
      auto outer_shared_arg = get<0>(shared_arg_tuple);
      auto inner_shared_arg = get<1>(shared_arg_tuple);

      using outer_shared_type = decltype(outer_shared_arg);
      using inner_shared_type = decltype(inner_shared_arg);

      // allocate outer shared argument
      // XXX need to pass outer_shared_arg
      auto outer_shared_arg_ptr = make_unique_cuda<outer_shared_type>();

      // copy construct the outer shared arg
      // XXX do this asynchronously
      //     don't do this if outer_shared_type is std::ignore
      bulk_invoke(detail::copy_outer_shared_parameter<outer_shared_type>(outer_shared_arg_ptr.get(), outer_shared_arg), shape_type{1,1});

      // wrap up f in a thing that will marshal the shared arguments to it
      // note the .release()
      auto g = detail::function_with_shared_arguments<Function, outer_shared_type, inner_shared_type>(f, outer_shared_arg_ptr.release(), inner_shared_arg);

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
      __throw_on_error(cudaDeviceSynchronize(), "cuda::grid_executor::bulk_invoke(): cudaDeviceSynchronize");
#  endif
#endif
    }


    template<class Function, class Tuple>
    __host__ __device__
    void bulk_invoke(Function f, shape_type shape, Tuple shared_arg_tuple)
    {
      auto outer_shared_arg = get<0>(shared_arg_tuple);
      auto inner_shared_arg = get<1>(shared_arg_tuple);

      using outer_shared_type = decltype(outer_shared_arg);
      using inner_shared_type = decltype(inner_shared_arg);

      // allocate outer shared argument
      // XXX need to pass outer_shared_arg
      auto outer_shared_arg_ptr = make_unique_cuda<outer_shared_type>();

      // copy construct the outer shared arg
      // XXX don't do this if outer_shared_type is std::ignore
      bulk_invoke(detail::copy_outer_shared_parameter<outer_shared_type>(outer_shared_arg_ptr.get(), outer_shared_arg), shape_type{1,1});

      // wrap up f in a thing that will marshal the shared arguments to it
      auto g = detail::function_with_shared_arguments<Function, outer_shared_type, inner_shared_type>(f, outer_shared_arg_ptr.get(), inner_shared_arg);

      return bulk_invoke(g, shape);
    }


    // this is exposed because it's necessary if a client wants to compute occupancy
    // alternatively, cuda_executor could report occupancy of a Function for a given block size
    template<class Function>
    __host__ __device__
    static decltype(&detail::grid_executor_kernel<Function>) global_function_pointer()
    {
      return &detail::grid_executor_kernel<Function>;
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

      detail::checked_launch_kernel_on_device(kernel, shape, shared_memory_size, stream, gpu.native_handle(), f);
    }

    int shared_memory_size_;
    cudaStream_t stream_;
    gpu_id gpu_;
};


template<class Function, class... Args>
__host__ __device__
void bulk_invoke(grid_executor& ex, typename grid_executor::shape_type shape, Function&& f, Args&&... args)
{
  auto g = thrust::experimental::bind(std::forward<Function>(f), thrust::placeholders::_1, std::forward<Args>(args)...);
  ex.bulk_invoke(g, shape);
}


namespace detail
{


template<class Function>
struct flattened_grid_executor_functor
{
  Function f_;
  std::size_t shape_;
  cuda::grid_executor::shape_type partitioning_;

  __host__ __device__
  flattened_grid_executor_functor(const Function& f, std::size_t shape, cuda::grid_executor::shape_type partitioning)
    : f_(f),
      shape_(shape),
      partitioning_(partitioning)
  {}

  template<class T>
  __device__
  void operator()(cuda::grid_executor::index_type idx, T&& shared_params)
  {
    auto flat_idx = get<0>(idx) * get<1>(partitioning_) + get<1>(idx);

    if(flat_idx < shape_)
    {
      f_(flat_idx, get<0>(shared_params));
    }
  }

  inline __device__
  void operator()(cuda::grid_executor::index_type idx)
  {
    auto flat_idx = get<0>(idx) * get<1>(partitioning_) + get<1>(idx);

    if(flat_idx < shape_)
    {
      f_(flat_idx);
    }
  }
};


} // end detail
} // end cuda


// specialize std::flattened_executor<grid_executor>
// to add __host__ __device__ to its functions and avoid lambdas
namespace std
{


template<>
class flattened_executor<cuda::grid_executor>
{
  public:
    using execution_category = parallel_execution_tag;
    using base_executor_type = cuda::grid_executor;

    // XXX maybe use whichever of the first two elements of base_executor_type::shape_type has larger dimensionality?
    using shape_type = size_t;

    // XXX initialize outer_subscription_ correctly
    __host__ __device__
    flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : outer_subscription_(2),
        base_executor_(base_executor)
    {}

    template<class Function>
    std::future<void> bulk_async(Function f, shape_type shape)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dummy_function, shape);

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return base_executor().bulk_async(execute_me, partitioning);
    }

    template<class Function, class T>
    std::future<void> bulk_async(Function f, shape_type shape, T shared_arg)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dummy_function, shape);

      // create a shared initializer
      auto shared_init = std::make_tuple(shared_arg, std::ignore);
      using shared_param_type = typename executor_traits<base_executor_type>::template shared_param_type<decltype(shared_init)>;

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return base_executor().bulk_async(execute_me, partitioning, shared_init);
    }

    __host__ __device__
    const base_executor_type& base_executor() const
    {
      return base_executor_;
    }

    __host__ __device__
    base_executor_type& base_executor()
    {
      return base_executor_;
    }

  private:

    using partition_type = typename executor_traits<base_executor_type>::shape_type;

    // returns (outer size, inner size)
    template<class Function>
    __host__ __device__
    partition_type partition(Function f, shape_type shape) const
    {
      using outer_shape_type = typename std::tuple_element<0,partition_type>::type;
      using inner_shape_type = typename std::tuple_element<1,partition_type>::type;

      auto max_shape = base_executor().max_shape(f);

      // make the inner groups as large as possible
      inner_shape_type inner_size = get<1>(max_shape);

      outer_shape_type outer_size = (shape + inner_size - 1) / inner_size;

      assert(outer_size <= get<0>(max_shape));

      return partition_type{outer_size, inner_size};
    }

    size_t min_inner_size_;
    size_t outer_subscription_;
    base_executor_type base_executor_;
};


} // end std

