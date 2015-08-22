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
#include <agency/detail/factory.hpp>
#include <agency/cuda/future.hpp>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class Factory>
struct result_of_factory_is_empty
  : std::integral_constant<
      bool,
      (std::is_empty<agency::detail::result_of_factory_t<Factory>>::value ||
      agency::detail::is_empty_tuple<agency::detail::result_of_factory_t<Factory>>::value)
    >
{};


template<class Factory, bool = result_of_factory_is_empty<Factory>::value>
struct outer_shared_parameter
{
  using value_type = agency::detail::result_of_factory_t<Factory>;

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


template<class Factory>
struct outer_shared_parameter<Factory,true>
{
  using value_type = agency::detail::result_of_factory_t<Factory>;

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


template<class Factory, bool = result_of_factory_is_empty<Factory>::value>
struct inner_shared_parameter
{
  using value_type = agency::detail::result_of_factory_t<Factory>;

  __device__
  inner_shared_parameter(bool is_leader, const Factory& factory)
    : is_leader_(is_leader)
  {
    __shared__ uninitialized<value_type> inner_shared_param;

    if(is_leader_)
    {
      // XXX should avoid the move construction here
      inner_shared_param.construct(factory());
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


template<class Factory>
struct inner_shared_parameter<Factory,true>
{
  using value_type = agency::detail::result_of_factory_t<Factory>;

  __device__
  inner_shared_parameter(bool is_leader_, const Factory&) {}

  inner_shared_parameter(const inner_shared_parameter&) = delete;
  inner_shared_parameter(inner_shared_parameter&&) = delete;

  __device__
  value_type& get()
  {
    return param_;
  }

  value_type param_;
};


template<class Function, class OuterFactory, class InnerFactory>
struct function_with_shared_arguments
{
  using outer_shared_type = agency::detail::result_of_factory_t<OuterFactory>;
  using inner_shared_type = agency::detail::result_of_factory_t<InnerFactory>;

  Function                             f_;
  outer_shared_parameter<OuterFactory> outer_shared_param_;
  InnerFactory                         inner_factory_;

  __host__ __device__
  function_with_shared_arguments(Function f, outer_shared_type* outer_shared_param_ptr, InnerFactory inner_factory)
    : f_(f),
      outer_shared_param_(outer_shared_param_ptr),
      inner_factory_(inner_factory)
  {}

  template<class Index, class... Args>
  __device__
  void operator()(const Index& idx, Args&&... args)
  {
    // XXX i don't think we're doing the leader calculation in a portable way
    //     this wouldn't work for ND CTAs
    inner_shared_parameter<InnerFactory> inner_shared_param(idx[1] == 0, inner_factory_);

    f_(idx, std::forward<Args>(args)..., outer_shared_param_.get(), inner_shared_param.get());
  }
};


template<class Function, class T>
struct function_with_past_parameter
{
  Function f_;
  T* past_param_ptr_;

  template<class Index>
  __device__
  void operator()(const Index& idx)
  {
    f_(idx, *past_param_ptr_);
  }
};


template<class ThisIndexFunction, class Function>
__global__ void grid_executor_kernel(Function f)
{
  f(ThisIndexFunction{}());
}


// XXX consider using unit_factory as the defaults, not void
//     we could probably simplify all of this stuff and not need any specialization here at all
template<class ThisIndexFunction, class Function, class PastParameterType = void, class OuterFactory = void, class InnerFactory = void>
struct global_function_pointer_map;


template<class ThisIndexFunction, class Function>
struct global_function_pointer_map<ThisIndexFunction,Function,void,void,void>
{
  using function_ptr_type = decltype(&grid_executor_kernel<ThisIndexFunction,Function>);

  __host__ __device__
  static function_ptr_type get()
  {
    return &grid_executor_kernel<ThisIndexFunction,Function>;
  }
};


template<class ThisIndexFunction, class Function, class PastParameterType>
struct global_function_pointer_map<ThisIndexFunction,Function,PastParameterType,void,void>
{
  using function_ptr_type = decltype(global_function_pointer_map<ThisIndexFunction, detail::function_with_past_parameter<Function,PastParameterType>>::get());

  __host__ __device__
  static function_ptr_type get()
  {
    return global_function_pointer_map<ThisIndexFunction, detail::function_with_past_parameter<Function,PastParameterType>>::get();
  }
};


template<class ThisIndexFunction, class Function, class PastParameterType, class OuterFactory, class InnerFactory>
struct global_function_pointer_map
{
  using function_ptr_type = decltype(global_function_pointer_map<ThisIndexFunction,detail::function_with_shared_arguments<Function,OuterFactory,InnerFactory>,PastParameterType>::get());

  __host__ __device__
  static function_ptr_type get()
  {
    return global_function_pointer_map<ThisIndexFunction, detail::function_with_shared_arguments<Function,OuterFactory,InnerFactory>, PastParameterType>::get();
  }
};


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

  private:
    template<class Function>
    __host__ __device__
    future<void> then_execute_impl(cudaEvent_t dependency, Function f, shape_type shape)
    {
      // XXX should we use this->stream() or the stream associated with the dependency?
      future<void> result{stream()};

      cudaEvent_t next_event = this->launch(global_function_pointer<Function>(), f, shape, shared_memory_size(), stream(), dependency);

      result.set_valid(next_event);

      return result;
    }

  public:
    template<class Function>
    __host__ __device__
    future<void> then_execute(future<void>& dependency, Function f, shape_type shape)
    {
      return then_execute_impl(dependency.event(), f, shape);
    }

    template<class T, class Function>
    __host__ __device__
    future<void> then_execute(future<T>& dependency, Function f, shape_type shape)
    {
      detail::function_with_past_parameter<Function,T> g{f, dependency.data()};

      // XXX we need to enqueue a destructor & deallocate for the dependency
      // XXX no, this should happen in future<T>'s destructor
      return then_execute_impl(dependency.event(), g, shape);
    }

  private:
    template<class T, class Function, class Factory1, class Factory2>
    __host__ __device__
    future<void> then_execute_with_shared_inits(future<T>& dependency, Function f, shape_type shape, Factory1 outer_factory, Factory2 inner_factory,
                                                typename std::enable_if<
                                                  result_of_factory_is_empty<Factory1>::value
                                                >::type* = 0)
    {
      // no need to marshal the outer arg through gmem because it is empty
      
      // wrap up f in a thing that will marshal the shared arguments to it
      // note the .release()
      auto g = detail::function_with_shared_arguments<Function, Factory1, Factory2>(f, 0, inner_factory);

      return this->then_execute(dependency, g, shape);
    }

    template<class T, class Function, class Factory1, class Factory2>
    __host__ __device__
    future<void> then_execute_with_shared_inits(future<T>& dependency, Function f, shape_type shape, Factory1 outer_factory, Factory2 inner_factory,
                                                typename std::enable_if<
                                                  !result_of_factory_is_empty<Factory1>::value
                                                >::type* = 0)
    {
      // XXX couldn't we implement outer_shared_init as a ready future?
      
      // XXX move call to factory inside of make_unique?
      auto outer_shared_arg = outer_factory();
      using outer_shared_arg_t = decltype(outer_shared_arg);

      // make outer shared argument
      auto outer_shared_arg_ptr = detail::make_unique<outer_shared_arg_t>(stream(), outer_shared_arg);

      // wrap up f in a thing that will marshal the shared arguments to it
      // note the .release()
      auto g = detail::function_with_shared_arguments<Function, Factory1, Factory2>(f, outer_shared_arg_ptr.release(), inner_factory);

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
    static typename detail::global_function_pointer_map<ThisIndexFunction, Function>::function_ptr_type
      global_function_pointer()
    {
      return detail::global_function_pointer_map<ThisIndexFunction, Function>::get();
    }


    template<class Function, class PastParameterType>
    __host__ __device__
    static typename detail::global_function_pointer_map<ThisIndexFunction, Function, PastParameterType>::function_ptr_type
      global_function_pointer()
    {
      return detail::global_function_pointer_map<ThisIndexFunction, Function, PastParameterType>::get();
    }


    template<class Function, class PastParameterType, class OuterFactory, class InnerFactory>
    __host__ __device__
    static typename detail::global_function_pointer_map<ThisIndexFunction, Function, PastParameterType, OuterFactory, InnerFactory>::function_ptr_type
      global_function_pointer()
    {
      return detail::global_function_pointer_map<ThisIndexFunction, Function, PastParameterType, OuterFactory, InnerFactory>::get();
    }


    template<class T, class Function, class OuterFactory, class InnerFactory>
    __host__ __device__
    future<void> then_execute(future<T>& dependency, Function f, shape_type shape, OuterFactory outer_factory, InnerFactory inner_factory)
    {
      return then_execute_with_shared_inits(dependency, f, shape, outer_factory, inner_factory);
    }


    // XXX we should eliminate these since executor_traits implements them for us
    template<class Function>
    __host__ __device__
    future<void> async_execute(Function f, shape_type shape)
    {
      auto ready = make_ready_future();
      return then_execute(ready, f, shape);
    }
    

    template<class Function, class Factory1, class Factory2>
    __host__ __device__
    future<void> async_execute(Function f, shape_type shape, Factory1 outer_factory, Factory2 inner_factory)
    {
      auto ready = make_ready_future();
      return then_execute(ready, f, shape, outer_factory, inner_factory);
    }


    template<class Function>
    __host__ __device__
    void execute(Function f, shape_type shape)
    {
      this->async_execute(f, shape).wait();
    }

    template<class Function, class Factory1, class Factory2>
    __host__ __device__
    void execute(Function f, shape_type shape, Factory1 outer_factory, Factory2 inner_factory)
    {
      this->async_execute(f, shape, outer_factory, inner_factory).wait();
    }


  private:
    template<class Arg>
    __host__ __device__
    cudaEvent_t launch(void (*kernel)(Arg), const Arg& arg, shape_type shape)
    {
      return launch(kernel, arg, shape, shared_memory_size());
    }

    template<class Arg>
    __host__ __device__
    cudaEvent_t launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size)
    {
      return launch(kernel, arg, shape, shared_memory_size, stream());
    }

    template<class Arg>
    __host__ __device__
    cudaEvent_t launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, cudaStream_t stream)
    {
      return launch(kernel, arg, shape, shared_memory_size, stream, 0);
    }

    template<class Arg>
    __host__ __device__
    cudaEvent_t launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, cudaStream_t stream, cudaEvent_t dependency)
    {
      return launch(kernel, arg, shape, shared_memory_size, stream, dependency, gpu());
    }

    template<class Arg>
    __host__ __device__
    cudaEvent_t launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, cudaStream_t stream, cudaEvent_t dependency, gpu_id gpu)
    {
      uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));
      uint3 inner_shape = agency::detail::shape_cast<uint3>(agency::detail::get<1>(shape));

      ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
      ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};

      return detail::checked_launch_kernel_after_event_on_device_returning_next_event(reinterpret_cast<void*>(kernel), grid_dim, block_dim, shared_memory_size, stream, dependency, gpu.native_handle(), arg);
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
    shape_type max_shape(const Function&) const
    {
      return max_shape_impl(reinterpret_cast<void*>(global_function_pointer<Function>()));
    }

    template<class T, class Function>
    __host__ __device__
    shape_type max_shape(const future<T>&, const Function&)
    {
      return max_shape_impl(reinterpret_cast<void*>(global_function_pointer<Function,T>()));
    }

    template<class T, class Function, class Factory1, class Factory2>
    __host__ __device__
    shape_type max_shape(const future<T>&, const Function&, const Factory1&, const Factory2&) const
    {
      return max_shape_impl(reinterpret_cast<void*>(global_function_pointer<Function,T,Factory1,Factory2>()));
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
  void operator()(cuda::grid_executor::index_type idx, T& outer_shared_param, agency::detail::unit)
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

    template<class T, class Function>
    future<void> then_execute(future<T>& dependency, Function f, shape_type shape)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dependency, dummy_function, shape);

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return base_executor().then_execute(dependency, execute_me, partitioning);
    }

    template<class T, class Function, class Factory>
    future<void> then_execute(future<T>& dependency, Function f, shape_type shape, Factory factory)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dependency, dummy_function, shape, factory, agency::detail::unit_factory());

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return base_executor().then_execute(dependency, execute_me, partitioning, factory, agency::detail::unit_factory());
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
    template<class T, class Function, class Factory1, class Factory2>
    __host__ __device__
    partition_type partition(const future<T>& dependency, const Function& f, shape_type shape, const Factory1& outer_factory, const Factory2& inner_factory) const
    {
      return partition_impl(base_executor().max_shape(dependency,f,outer_factory,inner_factory), shape);
    }

    size_t min_inner_size_;
    size_t outer_subscription_;
    base_executor_type base_executor_;
};


} // end agency

