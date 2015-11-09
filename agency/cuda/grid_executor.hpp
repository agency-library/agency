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
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/coordinate.hpp>
#include <agency/functional.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/factory.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/detail/array.hpp>


namespace agency
{
namespace cuda
{
namespace detail
{


// XXX should use empty base class optimization for this class because any of these members could be empty types
//     a simple way to apply this operation would be to derive this class from a tuple of its members, since tuple already applies EBO
// XXX should try to find a way to take an InnerParameterPointer instead of InnerFactory to make the way all the parameters are handled uniformly
// XXX the problem is that the inner parameter needs to know who the leader is, and that info isn't easily passed through pointer dereference syntax
template<class ContainerPointer, class Function, class IndexFunction, class PastParameterPointer, class OuterParameterPointer, class InnerFactory>
struct then_execute_functor {
  ContainerPointer      container_ptr_;
  Function              f_;
  IndexFunction         index_function_;
  PastParameterPointer  past_param_ptr_;
  OuterParameterPointer outer_param_ptr_;
  InnerFactory          inner_factory;

  // this gets called when the future we depend on is not void
  template<class Index, class T1, class T2, class T3, class T4>
  __device__ static inline void impl(Function f, const Index &idx, T1& container, T2& past_param, T3& outer_param, T4& inner_param)
  {
    container[idx] = f(idx, past_param, outer_param, inner_param);
  }

  // this gets called when the future we depend on is void
  template<class Index, class T1, class T3, class T4>
  __device__ static inline void impl(Function f, const Index &idx, T1& container, unit, T3& outer_param, T4& inner_param)
  {
    container[idx] = f(idx, outer_param, inner_param);
  }
  
  __device__ inline void operator()()
  {
    // we need to cast each dereference below to convert proxy references to ensure that f() only sees raw references
    // XXX isn't there a more elegant way to deal with this?
    using container_reference   = typename std::pointer_traits<ContainerPointer>::element_type &;
    using past_param_reference  = typename std::pointer_traits<PastParameterPointer>::element_type &;
    using outer_param_reference = typename std::pointer_traits<OuterParameterPointer>::element_type &;

    auto idx = index_function_();

    // XXX i don't think we're doing the leader calculation in a portable way
    //     we need a way to compare idx to the origin idex to figure out if this invocation represents the CTA leader
    on_chip_shared_parameter<InnerFactory> inner_param(idx[1] == 0, inner_factory);

    impl(
      f_,
      idx,
      static_cast<container_reference>(*container_ptr_),
      static_cast<past_param_reference>(*past_param_ptr_),
      static_cast<outer_param_reference>(*outer_param_ptr_),
      inner_param.get()
    );
  }
};


template<class ContainerPointer, class Function, class IndexFunction, class PastParameterPointer, class OuterParameterPointer, class InnerFactory>
__host__ __device__
then_execute_functor<ContainerPointer,Function,IndexFunction,PastParameterPointer,OuterParameterPointer,InnerFactory>
  make_then_execute_functor(ContainerPointer container_ptr, Function f, IndexFunction index_function, PastParameterPointer past_param_ptr, OuterParameterPointer outer_param_ptr, InnerFactory inner_factory)
{
  return then_execute_functor<ContainerPointer,Function,IndexFunction,PastParameterPointer,OuterParameterPointer,InnerFactory>{container_ptr, f, index_function, past_param_ptr, outer_param_ptr, inner_factory};
}


template<class Function>
__global__ void grid_executor_kernel(Function f)
{
  f();
}


// this computes the type of the result returned by basic_grid_executor::then_execute_kernel_impl()
// XXX eliminate this
template<class Container, class Function, class IndexFunction, class PastParameterType, class OuterFactory, class InnerFactory>
struct then_execute_kernel_impl_result
{
  // XXX these types need to agree the way then_execute() creates the then_execute_functor
  // XXX there must be a sounder way to do this stuff

  using container_ptr_type = decltype(std::declval<future<Container>>().data());
  using past_parameter_ptr_type = decltype(std::declval<future<PastParameterType>>().data());

  using outer_arg_type = agency::detail::result_of_factory_t<OuterFactory>;
  using outer_parameter_ptr_type = decltype(std::declval<future<outer_arg_type>>().data());

  using functor_type = then_execute_functor<container_ptr_type, Function, IndexFunction, past_parameter_ptr_type, outer_parameter_ptr_type, InnerFactory>;

  using type = decltype(&detail::grid_executor_kernel<functor_type>);
};


template<class Function, class FutureValueType>
struct take_first_two_parameters_and_invoke : agency::detail::take_first_two_parameters_and_invoke<Function>
{
};


template<class Function>
struct take_first_two_parameters_and_invoke<Function,void> : agency::detail::take_first_parameter_and_invoke<Function>
{
};


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


    template<class T>
    using container = detail::array<T, shape_type>;


    __host__ __device__
    explicit basic_grid_executor(int shared_memory_size = 0, gpu_id gpu = detail::current_gpu())
      : shared_memory_size_(shared_memory_size),
        gpu_(gpu)
    {}


    __host__ __device__
    int shared_memory_size() const
    {
      return shared_memory_size_;
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
    // XXX we should push this call into then_execute() since there's no longer a reason to have this extra function
    template<class Function>
    __host__ __device__
    event then_execute_impl(Function f, shape_type shape, stream& stream, event& dependency)
    {
      // XXX shouldn't we use the stream associated with dependency instead of stream()?

      return this->launch(grid_executor_kernel<Function>, f, shape, shared_memory_size(), stream, dependency);
    }


    // this returns a pointer to the __global__ function used to implement then_execute()
    template<class Container, class Function, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    static typename then_execute_kernel_impl_result<
        Container, Function, ThisIndexFunction, T, OuterFactory, InnerFactory
    >::type
      then_execute_kernel_impl(const Function&, const future<T>&, const OuterFactory&, const InnerFactory&)
    {
      // XXX these types need to agree the way then_execute() creates the then_execute_functor
      // XXX there must be a sounder way to do this stuff

      using container_ptr_type = decltype(std::declval<future<Container>>().data());
      using past_parameter_ptr_type = decltype(std::declval<future<T>>().data());

      using outer_arg_type = agency::detail::result_of_factory_t<OuterFactory>;
      using outer_parameter_ptr_type = decltype(std::declval<future<outer_arg_type>>().data());

      using functor_type = then_execute_functor<container_ptr_type, Function, ThisIndexFunction, past_parameter_ptr_type, outer_parameter_ptr_type, InnerFactory>;

      return &detail::grid_executor_kernel<functor_type>;
    }

  public:
    template<class Container, class Function, class T, class Factory1, class Factory2,
             class = typename std::enable_if<
               agency::detail::new_executor_traits_detail::is_container<Container,index_type>::value
             >::type,
             class = agency::detail::result_of_continuation_t<
               Function,
               index_type,
               future<T>,
               agency::detail::result_of_factory_t<Factory1>&,
               agency::detail::result_of_factory_t<Factory2>&
             >
            >
    __host__ __device__
    future<Container> then_execute(Function f, shape_type shape, future<T>& fut, Factory1 outer_factory, Factory2 inner_factory)
    {
      detail::stream stream = std::move(fut.stream());

      detail::asynchronous_state<Container> result_state(construct_ready, shape);

      using outer_arg_type = agency::detail::result_of_factory_t<Factory1>;
      auto outer_arg = executor_traits<basic_grid_executor>::template make_ready_future<outer_arg_type>(*this, outer_factory());

      auto g = make_then_execute_functor(result_state.data(), f, ThisIndexFunction(), fut.data(), outer_arg.data(), inner_factory);

      auto next_event = then_execute_impl(g, shape, stream, fut.event());

      return future<Container>(std::move(stream), std::move(next_event), std::move(result_state));
    }


    // this is exposed because it's necessary if a client wants to compute occupancy
    // alternatively, cuda_executor could report occupancy of a Function for a given block size
    // XXX probably shouldn't expose this -- max_shape() seems good enough

    template<class Container, class Function, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    static auto then_execute_kernel(const Function& f, const future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory)
      -> decltype(
           then_execute_kernel_impl<Container>(f, fut, outer_factory, inner_factory)
         )
    {
      return then_execute_kernel_impl<Container>(f, fut, outer_factory, inner_factory);
    }


    template<class Function, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    static auto then_execute_kernel(const Function& f, const future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory)
      -> decltype(
           then_execute_kernel_impl<agency::detail::new_executor_traits_detail::discarding_container>(
             agency::detail::new_executor_traits_detail::invoke_and_return_empty<Function>{f}, // XXX replace with invoke_and_return_unit
             fut,
             outer_factory,
             inner_factory
           )
         )
    {
      // XXX these types need to agree the way executor_traits lowers multi-agent then_execute() with shared inits returning void onto
      //     the more general version of then_execute()
      // XXX there must be a sounder way to do this stuff
      return then_execute_kernel_impl<agency::detail::new_executor_traits_detail::discarding_container>(
        agency::detail::new_executor_traits_detail::invoke_and_return_empty<Function>{f}, // XXX replace with invoke_and_return_unit
        fut,
        outer_factory,
        inner_factory
      );
    }

    template<class Function, class T>
    __host__ __device__
    static auto then_execute_kernel(const Function& f, const future<T>& fut)
      -> decltype(
           then_execute_kernel(detail::take_first_two_parameters_and_invoke<Function,T>{f}, fut, agency::detail::unit_factory(), agency::detail::unit_factory())
         )
    {
      return then_execute_kernel(detail::take_first_two_parameters_and_invoke<Function,T>{f}, fut, agency::detail::unit_factory(), agency::detail::unit_factory());
    }


    template<class Function>
    __host__ __device__
    static auto then_execute_kernel(const Function& f)
      -> decltype(then_execute_kernel(f, future<void>()))
    {
      return then_execute_kernel(f, future<void>());
    }


    // XXX we should eliminate these functions below since executor_traits implements them for us

    template<class Function>
    __host__ __device__
    future<void> async_execute(Function f, shape_type shape)
    {
      auto ready = make_ready_future();
      return agency::executor_traits<basic_grid_executor>::then_execute(*this, f, shape, ready);
    }
    

    template<class Function, class Factory1, class Factory2>
    __host__ __device__
    future<void> async_execute(Function f, shape_type shape, Factory1 outer_factory, Factory2 inner_factory)
    {
      auto ready = make_ready_future();
      return agency::executor_traits<basic_grid_executor>::then_execute(*this, f, shape, ready, outer_factory, inner_factory);
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
    event launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, stream& stream, event& dependency)
    {
      return launch(kernel, arg, shape, shared_memory_size, stream, dependency, gpu());
    }

    template<class Arg>
    __host__ __device__
    event launch(void (*kernel)(Arg), const Arg& arg, shape_type shape, int shared_memory_size, stream& stream, event& dependency, gpu_id gpu)
    {
      uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));
      uint3 inner_shape = agency::detail::shape_cast<uint3>(agency::detail::get<1>(shape));

      ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
      ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};

      return dependency.then_on(reinterpret_cast<void*>(kernel), grid_dim, block_dim, shared_memory_size, stream.native_handle(), gpu.native_handle(), arg);
    }

    int shared_memory_size_;
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

      int max_grid_dimension_x = 0;
      detail::throw_on_error(cudaDeviceGetAttribute(&max_grid_dimension_x, cudaDevAttrMaxGridDimX, gpu().native_handle()),
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

      result = shape_type{static_cast<unsigned int>(max_grid_dimension_x), static_cast<unsigned int>(attr.maxThreadsPerBlock)};
#endif // __cuda_lib_has_cudart

      return result;
    }

  public:
    template<class Function>
    __host__ __device__
    shape_type max_shape(const Function& f) const
    {
      return max_shape_impl(reinterpret_cast<void*>(then_execute_kernel(f)));
    }

    template<class Function, class T>
    __host__ __device__
    shape_type max_shape(const Function& f, const future<T>& fut)
    {
      return max_shape_impl(reinterpret_cast<void*>(then_execute_kernel(f,fut)));
    }

    template<class T, class Function, class Factory1, class Factory2>
    __host__ __device__
    shape_type max_shape(const Function& f, const future<T>& fut, const Factory1& outer_factory, const Factory2& inner_factory) const
    {
      return max_shape_impl(reinterpret_cast<void*>(then_execute_kernel(f, fut, outer_factory, inner_factory)));
    }
};


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

    template<class Function, class T>
    future<void> then_execute(Function f, shape_type shape, future<T>& dependency)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dependency, dummy_function, shape);

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return base_executor().then_execute(execute_me, partitioning, dependency);
    }

    template<class Function, class T, class Factory>
    future<void> then_execute(Function f, shape_type shape, future<T>& dependency, Factory factory)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partition_type{}};

      // partition up the iteration space
      auto partitioning = partition(dependency, dummy_function, shape, factory, agency::detail::unit_factory());

      // create a function to execute
      auto execute_me = cuda::detail::flattened_grid_executor_functor<Function>{f, shape, partitioning};

      return agency::executor_traits<base_executor_type>::then_execute(base_executor(), execute_me, partitioning, dependency, factory, agency::detail::unit_factory());
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
      return partition_impl(base_executor().max_shape(f,dependency,outer_factory,inner_factory), shape);
    }

    size_t min_inner_size_;
    size_t outer_subscription_;
    base_executor_type base_executor_;
};


} // end agency

