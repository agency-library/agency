#pragma once

#include <agency/execution_categories.hpp>
#include <memory>
#include <iostream>
#include <exception>
#include <cstring>
#include <type_traits>
#include <cassert>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <agency/flattened_executor.hpp>
#include <agency/detail/tuple.hpp>
#include <thrust/detail/minmax.h>
#include <agency/cuda/detail/tuple.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/gpu.hpp>
#include <agency/cuda/detail/bind.hpp>
#include <agency/cuda/detail/unique_ptr.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/uninitialized.hpp>
#include <agency/cuda/detail/launch_kernel.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/coordinate.hpp>
#include <agency/functional.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/cuda/future.hpp>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
struct is_ignorable_shared_parameter
  : std::integral_constant<
      bool,
      (std::is_empty<decay_construct_result_t<T>>::value || agency::detail::is_empty_tuple<decay_construct_result_t<T>>::value)
    >
{};


template<class InitT, bool = is_ignorable_shared_parameter<InitT>::value>
struct outer_shared_parameter
{
  using value_type = decay_construct_result_t<InitT>;

  __device__
  outer_shared_parameter(value_type* outer_shared_param_ptr)
    : param_(*outer_shared_param_ptr)
  {}

  __device__
  value_type& get()
  {
    return param_;
  }

  value_type& param_;
};


template<class InitT>
struct outer_shared_parameter<InitT,true>
{
  using value_type = decay_construct_result_t<InitT>;

  __device__
  outer_shared_parameter(value_type*) {}

  __device__
  value_type& get()
  {
    return param_;
  }

  // the parameter is "ignorable" so we needn't store a reference to an object in global memory
  value_type param_;
};


template<class InitT, bool = is_ignorable_shared_parameter<InitT>::value>
struct inner_shared_parameter
{
  using value_type = decay_construct_result_t<InitT>;

  __device__
  inner_shared_parameter(bool is_leader, const InitT& init)
    : is_leader_(is_leader)
  {
    __shared__ uninitialized<value_type> inner_shared_param;

    if(is_leader_)
    {
      // XXX should avoid the move construction here
      inner_shared_param.construct(agency::decay_construct(init));
    }
    __syncthreads();

    inner_shared_param_ = &inner_shared_param;
  }

  inner_shared_parameter(const inner_shared_parameter&) = delete;
  inner_shared_parameter(inner_shared_parameter&&) = delete;

  __device__
  ~inner_shared_parameter()
  {
    __syncthreads();

    if(is_leader_)
    {
      inner_shared_param_->destroy();
    }
  }

  __device__
  value_type& get()
  {
    return inner_shared_param_->get();
  }

  const bool is_leader_;
  uninitialized<value_type>* inner_shared_param_;
};


template<class InitT>
struct inner_shared_parameter<InitT,true>
{
  using value_type = decay_construct_result_t<InitT>;

  __device__
  inner_shared_parameter(bool is_leader_, const InitT& init) {}

  inner_shared_parameter(const inner_shared_parameter&) = delete;
  inner_shared_parameter(inner_shared_parameter&&) = delete;

  __device__
  value_type& get()
  {
    return param_;
  }

  value_type param_;
};


template<class Function, class OuterSharedInit, class InnerSharedInit>
struct function_with_shared_arguments
{
  using outer_shared_type = decay_construct_result_t<OuterSharedInit>;
  using inner_shared_type = decay_construct_result_t<InnerSharedInit>;

  Function           f_;
  outer_shared_parameter<OuterSharedInit> outer_shared_param_;
  InnerSharedInit                         inner_shared_init_;

  __host__ __device__
  function_with_shared_arguments(Function f, outer_shared_type* outer_shared_param_ptr, InnerSharedInit inner_shared_init)
    : f_(f),
      outer_shared_param_(outer_shared_param_ptr),
      inner_shared_init_(inner_shared_init)
  {}

  template<class Index>
  __device__
  void operator()(const Index& idx)
  {
    // XXX i don't think we're doing the leader calculation in a portable way
    //     this wouldn't work for ND CTAs
    inner_shared_parameter<InnerSharedInit> inner_shared_param(idx[1] == 0, inner_shared_init_);

    f_(idx, outer_shared_param_.get(), inner_shared_param.get());
  }
};


template<class ThisIndexFunction, class Function>
__global__ void grid_executor_kernel(Function f)
{
  f(ThisIndexFunction{}());
}


void grid_executor_notify(cudaStream_t stream, cudaError_t status, void* data)
{
  std::unique_ptr<std::promise<void>> promise(reinterpret_cast<std::promise<void>*>(data));

  promise->set_value();
}


template<class Shape, class Index, class ThisIndexFunction>
class basic_grid_executor
{
  public:
    using execution_category =
      nested_execution_tag<
        parallel_execution_tag,
        concurrent_execution_tag
      >;


    using shape_type = Shape;


    using index_type = Index;


    template<class T>
    using future = cuda::future<T>;


    __host__ __device__
    explicit basic_grid_executor(int shared_memory_size = 0, cudaStream_t stream = 0, gpu_id gpu = detail::current_gpu())
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

    
    __host__ __device__
    future<void> make_ready_future()
    {
      return cuda::make_ready_future();
    }


    template<class Function>
    __host__ __device__
    future<void> then_execute(future<void>& dependency, Function f, shape_type shape)
    {
      this->launch(global_function_pointer(f), f, shape, shared_memory_size(), stream(), dependency.event());

      return future<void>{stream()};
    }

  private:
    template<class Function, class T1, class T2>
    __host__ __device__
    future<void> then_execute_with_shared_inits(future<void>& dependency, Function f, shape_type shape, const T1& outer_shared_init, const T2& inner_shared_init,
                                                typename std::enable_if<
                                                  is_ignorable_shared_parameter<T1>::value
                                                >::type* = 0)
    {
      // no need to marshal the outer arg through gmem because it is empty
      
      // wrap up f in a thing that will marshal the shared arguments to it
      // note the .release()
      auto g = detail::function_with_shared_arguments<Function, T1, T2>(f, 0, inner_shared_init);

      return this->then_execute(dependency, g, shape);
    }

    template<class Function, class T1, class T2>
    __host__ __device__
    future<void> then_execute_with_shared_inits(future<void>& dependency, Function f, shape_type shape, const T1& outer_shared_init, const T2& inner_shared_init,
                                                typename std::enable_if<
                                                  !is_ignorable_shared_parameter<T1>::value
                                                >::type* = 0)
    {
      // XXX move decay_construct inside of make_unique
      auto outer_shared_arg = agency::decay_construct(outer_shared_init);
      using outer_shared_arg_t = typename decay_construct_result<T1>::type;

      // make outer shared argument
      auto outer_shared_arg_ptr = detail::make_unique<outer_shared_arg_t>(stream(), outer_shared_arg);

      // wrap up f in a thing that will marshal the shared arguments to it
      // note the .release()
      auto g = detail::function_with_shared_arguments<Function, T1, T2>(f, outer_shared_arg_ptr.release(), inner_shared_init);

      // XXX to destroy the outer_shared_arg, we need to do another then_execute(...)
      // XXX to deallocate the outer_shared_arg's storage, we need to enqueue a free after the 2nd then_execute somehow
      // XXX for now the memory just leaks :(

      return this->then_execute(dependency, g, shape);
    }

  public:
    // this is exposed because it's necessary if a client wants to compute occupancy
    // alternatively, cuda_executor could report occupancy of a Function for a given block size
    // XXX probably shouldn't expose this -- max_shape() seems good enough
    template<class Function>
    __host__ __device__
    static decltype(&detail::grid_executor_kernel<ThisIndexFunction,Function>)
      global_function_pointer(const Function&)
    {
      return &detail::grid_executor_kernel<ThisIndexFunction, Function>;
    }


    template<class Function, class T1, class T2>
    __host__ __device__
    static decltype(
      &detail::grid_executor_kernel<
        ThisIndexFunction,
        detail::function_with_shared_arguments<Function, T1, T2>
      >
    )
      global_function_pointer(const Function&, const T1&, const T2&)
    {
      return &detail::grid_executor_kernel<
        ThisIndexFunction,
        detail::function_with_shared_arguments<Function,T1,T2>
      >;
    }


    template<class Function, class T1, class T2>
    __host__ __device__
    future<void> then_execute(future<void>& dependency, Function f, shape_type shape, T1&& outer_shared_init, T2&& inner_shared_init)
    {
      return then_execute_with_shared_inits(dependency, f, shape, std::forward<T1>(outer_shared_init), std::forward<T2>(inner_shared_init));
    }


    // XXX we should eliminate these since executor_traits implements them for us
    template<class Function>
    __host__ __device__
    future<void> async_execute(Function f, shape_type shape)
    {
      auto ready = make_ready_future();
      return then_execute(ready, f, shape);
    }
    

    template<class Function, class T1, class T2>
    __host__ __device__
    future<void> async_execute(Function f, shape_type shape, T1&& outer_shared_init, T2&& inner_shared_init)
    {
      auto ready = make_ready_future();
      return then_execute(ready, f, shape, std::forward<T1>(outer_shared_init), std::forward<T2>(inner_shared_init));
    }


    template<class Function>
    __host__ __device__
    void execute(Function f, shape_type shape)
    {
      this->async_execute(f, shape).wait();
    }

    template<class Function, class T1, class T2>
    __host__ __device__
    void execute(Function f, shape_type shape, T1&& outer_shared_init, T2&& inner_shared_init)
    {
      this->async_execute(f, shape, std::forward<T1>(outer_shared_init), std::forward<T2>(inner_shared_init)).wait();
    }


  private:
    template<class Arg>
    __host__ __device__
    void launch(void (*kernel)(Arg), const Arg& arg, shape_type shape)
    {
      launch(kernel, arg, shape, shared_memory_size());
    }

    template<class Arg>
    __host__ __device__
    void launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size)
    {
      launch(kernel, arg, shape, shared_memory_size, stream());
    }

    template<class Arg>
    __host__ __device__
    void launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, cudaStream_t stream)
    {
      launch(kernel, arg, shape, shared_memory_size, stream, 0);
    }

    template<class Arg>
    __host__ __device__
    void launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, cudaStream_t stream, cudaEvent_t dependency)
    {
      launch(kernel, arg, shape, shared_memory_size, stream, dependency, gpu());
    }

    template<class Arg>
    __host__ __device__
    void launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, cudaStream_t stream, cudaEvent_t dependency, gpu_id gpu)
    {
      uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));
      uint3 inner_shape = agency::detail::shape_cast<uint3>(agency::detail::get<1>(shape));

      ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
      ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};

      detail::checked_launch_kernel_after_event_on_device(reinterpret_cast<void*>(kernel), grid_dim, block_dim, shared_memory_size, stream, dependency, gpu.native_handle(), arg);
    }

    int shared_memory_size_;
    cudaStream_t stream_;
    gpu_id gpu_;
};


struct this_index_1d
{
  __device__
  agency::uint2 operator()() const
  {
    return agency::uint2{blockIdx.x, threadIdx.x};
  }
};


struct this_index_2d
{
  __device__
  agency::point<agency::uint2,2> operator()() const
  {
    auto block = agency::uint2{blockIdx.x, blockIdx.y};
    auto thread = agency::uint2{threadIdx.x, threadIdx.y};
    return agency::point<agency::uint2,2>{block, thread};
  }
};


} // end detail


class grid_executor : public detail::basic_grid_executor<agency::uint2, agency::uint2, detail::this_index_1d>
{
  public:
    using detail::basic_grid_executor<agency::uint2, agency::uint2, detail::this_index_1d>::basic_grid_executor;

  private:
    inline __host__ __device__
    shape_type max_shape_impl(void* fun_ptr) const
    {
      shape_type result = {0,0};

      detail::workaround_unused_variable_warning(fun_ptr);

#if __cuda_lib_has_cudart
      // record the current device
      int current_device = 0;
      detail::throw_on_error(cudaGetDevice(&current_device), "cuda::grid_executor::max_shape(): cudaGetDevice()");
      if(current_device != gpu().native_handle())
      {
#  ifndef __CUDA_ARCH__
        detail::throw_on_error(cudaSetDevice(gpu().native_handle()), "cuda::grid_executor::max_shape(): cudaSetDevice()");
#  else
        detail::throw_on_error(cudaErrorNotSupported, "cuda::grid_executor::max_shape(): cudaSetDevice only allowed in __host__ code");
#  endif // __CUDA_ARCH__
      }

      int max_block_dimension_x = 0;
      detail::throw_on_error(cudaDeviceGetAttribute(&max_block_dimension_x, cudaDevAttrMaxBlockDimX, gpu().native_handle()),
                             "cuda::grid_executor::max_shape(): cudaDeviceGetAttribute");

      cudaFuncAttributes attr{};
      detail::throw_on_error(cudaFuncGetAttributes(&attr, fun_ptr),
                             "cuda::grid_executor::max_shape(): cudaFuncGetAttributes");

      // restore current device
      if(current_device != gpu().native_handle())
      {
#  ifndef __CUDA_ARCH__
        detail::throw_on_error(cudaSetDevice(current_device), "cuda::grid_executor::max_shape(): cudaSetDevice()");
#  else
        detail::throw_on_error(cudaErrorNotSupported, "cuda::grid_executor::max_shape(): cudaSetDevice only allowed in __host__ code");
#  endif // __CUDA_ARCH__
      }

      result = shape_type{static_cast<unsigned int>(max_block_dimension_x), static_cast<unsigned int>(attr.maxThreadsPerBlock)};
#endif // __cuda_lib_has_cudart

      return result;
    }

  public:
    template<class Function>
    __host__ __device__
    shape_type max_shape(const Function& f) const
    {
      return max_shape_impl(reinterpret_cast<void*>(global_function_pointer(f)));
    }

    template<class Function, class T1, class T2>
    __host__ __device__
    shape_type max_shape(const Function& f, const T1& outer_shared_init, const T2& inner_shared_init) const
    {
      return max_shape_impl(reinterpret_cast<void*>(global_function_pointer(f, outer_shared_init, inner_shared_init)));
    }
};


// XXX eliminate this
template<class Function, class... Args>
__host__ __device__
void bulk_invoke(grid_executor& ex, typename grid_executor::shape_type shape, Function&& f, Args&&... args)
{
  auto g = detail::bind(std::forward<Function>(f), detail::placeholders::_1, std::forward<Args>(args)...);
  ex.execute(g, shape);
}


class grid_executor_2d : public detail::basic_grid_executor<
  point<agency::uint2,2>,
  point<agency::uint2,2>,
  detail::this_index_2d
>
{
  public:
    using detail::basic_grid_executor<
      point<agency::uint2,2>,
      point<agency::uint2,2>,
      detail::this_index_2d
    >::basic_grid_executor;

    // XXX implement max_shape()
};


// XXX eliminate this
template<class Function, class... Args>
__host__ __device__
void bulk_invoke(grid_executor_2d& ex, typename grid_executor_2d::shape_type shape, Function&& f, Args&&... args)
{
  auto g = detail::bind(std::forward<Function>(f), detail::placeholders::_1, std::forward<Args>(args)...);
  ex.execute(g, shape);
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
  void operator()(cuda::grid_executor::index_type idx, T& outer_shared_param, const agency::detail::ignore_t&)
  {
    auto flat_idx = agency::detail::get<0>(idx) * agency::detail::get<1>(partitioning_) + agency::detail::get<1>(idx);

    if(flat_idx < shape_)
    {
      f_(flat_idx, outer_shared_param);
    }
  }

  inline __device__
  void operator()(cuda::grid_executor::index_type idx)
  {
    auto flat_idx = agency::detail::get<0>(idx) * agency::detail::get<1>(partitioning_) + agency::detail::get<1>(idx);

    if(flat_idx < shape_)
    {
      f_(flat_idx);
    }
  }
};


} // end detail
} // end cuda


// specialize agency::flattened_executor<grid_executor>
// to add __host__ __device__ to its functions and avoid lambdas
template<>
class flattened_executor<cuda::grid_executor>
{
  public:
    using execution_category = parallel_execution_tag;
    using base_executor_type = cuda::grid_executor;

    // XXX maybe use whichever of the first two elements of base_executor_type::shape_type has larger dimensionality?
    using shape_type = size_t;

    template<class T>
    using future = cuda::grid_executor::template future<T>;

    // XXX initialize outer_subscription_ correctly
    __host__ __device__
    flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : outer_subscription_(2),
        base_executor_(base_executor)
    {}

    template<class Function>
    future<void> then_execute(future<void>& dependency, Function f, shape_type shape)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dummy_function, shape);

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return base_executor().then_execute(dependency, execute_me, partitioning);
    }

    template<class Function, class T>
    future<void> then_execute(future<void>& dependency, Function f, shape_type shape, T shared_init)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dummy_function, shape, shared_init, agency::detail::ignore);

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return base_executor().then_execute(dependency, execute_me, partitioning, shared_init, agency::detail::ignore);
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

    __host__ __device__
    static partition_type partition_impl(partition_type max_shape, shape_type shape)
    {
      using outer_shape_type = typename std::tuple_element<0,partition_type>::type;
      using inner_shape_type = typename std::tuple_element<1,partition_type>::type;

      // make the inner groups as large as possible
      inner_shape_type inner_size = agency::detail::get<1>(max_shape);

      outer_shape_type outer_size = (shape + inner_size - 1) / inner_size;

      assert(outer_size <= agency::detail::get<0>(max_shape));

      return partition_type{outer_size, inner_size};
    }

    // returns (outer size, inner size)
    template<class Function>
    __host__ __device__
    partition_type partition(const Function& f, shape_type shape) const
    {
      return partition_impl(base_executor().max_shape(f), shape);
    }

    // returns (outer size, inner size)
    template<class Function, class T1, class T2>
    __host__ __device__
    partition_type partition(const Function& f, shape_type shape, const T1& outer_shared_init, const T2& inner_shared_init) const
    {
      return partition_impl(base_executor().max_shape(f,outer_shared_init,inner_shared_init), shape);
    }

    size_t min_inner_size_;
    size_t outer_subscription_;
    base_executor_type base_executor_;
};


} // end agency

