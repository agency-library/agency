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
#include <agency/cuda/detail/memory/unique_ptr.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/cuda/detail/when_all_execute_and_select.hpp>
#include <agency/cuda/detail/then_execute.hpp>
#include <agency/cuda/detail/new_then_execute.hpp>
#include <agency/coordinate.hpp>
#include <agency/functional.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/factory.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/detail/array.hpp>
#include <agency/detail/optional.hpp>
#include <agency/detail/flatten_index_and_invoke.hpp>


namespace agency
{
namespace cuda
{
namespace detail
{


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
    explicit basic_grid_executor(gpu_id gpu = detail::current_gpu())
      : gpu_(gpu)
    {}


    __host__ __device__
    gpu_id gpu() const
    {
      return gpu_;
    }

    
    __host__ __device__
    void gpu(gpu_id gpu)
    {
      gpu_ = gpu;
    }

    
    __host__ __device__
    future<void> make_ready_future()
    {
      return cuda::make_ready_future();
    }

    template<size_t... Indices, class Function, class TupleOfFutures, class Factory1, class Factory2>
    __host__ __device__
    future<detail::when_all_execute_and_select_result_t<agency::detail::index_sequence<Indices...>, agency::detail::decay_t<TupleOfFutures>>>
      when_all_execute_and_select(Function f,
                                  shape_type shape,
                                  TupleOfFutures&& tuple_of_futures,
                                  Factory1 outer_factory,
                                  Factory2 inner_factory)
    {
      return detail::when_all_execute_and_select<Indices...>(f, shape, ThisIndexFunction(), std::forward<TupleOfFutures>(tuple_of_futures), outer_factory, inner_factory, gpu());
    }

    template<class Function, class Factory1, class T, class Factory2, class Factory3,
             class = agency::detail::result_of_continuation_t<
               Function,
               index_type,
               future<T>,
               agency::detail::result_of_factory_t<Factory2>&,
               agency::detail::result_of_factory_t<Factory3>&
             >
            >
    __host__ __device__
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, future<T>& fut, Factory2 outer_factory, Factory3 inner_factory)
    {
      return detail::new_then_execute(f, result_factory, shape, ThisIndexFunction(), fut, outer_factory, inner_factory, gpu());
    }


    // this is exposed because it's necessary if a client wants to compute occupancy
    // alternatively, cuda_executor could report occupancy of a Function for a given block size
    // XXX probably shouldn't expose this -- max_shape() seems good enough

    template<class Container, class Function, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    void* then_execute_kernel(const Function& f, const future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory) const
    {
      return detail::then_execute_kernel<Container>(f, shape_type{}, ThisIndexFunction(), fut, outer_factory, inner_factory);
    }


    template<class Function, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    void* then_execute_kernel(const Function& f, const future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory) const
    {
      using container_type = agency::detail::new_executor_traits_detail::discarding_container;
      auto g = agency::detail::invoke_and_return_unit<Function>{f};
      return detail::then_execute_kernel<container_type>(g, shape_type{}, ThisIndexFunction(), fut, outer_factory, inner_factory);
    }

    template<class Function, class T>
    __host__ __device__
    static void* then_execute_kernel(const Function& f, const future<T>& fut)
    {
      using container_type = agency::detail::new_executor_traits_detail::discarding_container;
      auto g = agency::detail::invoke_and_return_unit<Function>{f};
      return detail::then_execute_kernel<container_type>(g, shape_type{}, ThisIndexFunction(), fut);
    }


    template<class Function>
    __host__ __device__
    static void* then_execute_kernel(const Function& f)
    {
      return then_execute_kernel(f, future<void>());
    }


    template<class Function, class Factory1, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    void* new_then_execute_kernel(const Function& f, const Factory1& result_factory, const future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory) const
    {
      return detail::new_then_execute_kernel(f, result_factory, shape_type{}, ThisIndexFunction(), fut, outer_factory, inner_factory, gpu());
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
      return max_shape_impl(then_execute_kernel(f));
    }

    template<class Function, class T>
    __host__ __device__
    shape_type max_shape(const Function& f, const future<T>& fut)
    {
      return max_shape_impl(then_execute_kernel(f,fut));
    }

    template<class T, class Function, class Factory1, class Factory2>
    __host__ __device__
    shape_type max_shape(const Function& f, const future<T>& fut, const Factory1& outer_factory, const Factory2& inner_factory) const
    {
      return max_shape_impl(then_execute_kernel(f, fut, outer_factory, inner_factory));
    }

    template<class Function, class Factory1, class T, class Factory2, class Factory3>
    __host__ __device__
    shape_type max_shape(const Function& f, const Factory1& result_factory, const future<T>& fut, const Factory2& outer_factory, const Factory3& inner_factory) const
    {
      return max_shape_impl(new_then_execute_kernel(f, result_factory, fut, outer_factory, inner_factory));
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


template<class Container>
struct guarded_container : Container
{
  using Container::Container;

  __host__ __device__
  guarded_container(Container&& other)
    : Container(std::move(other))
  {}

  struct reference
  {
    Container& self_;

    template<class OptionalValueAndIndex>
    __host__ __device__
    void operator=(OptionalValueAndIndex&& opt)
    {
      if(opt)
      {
        auto idx = opt.value().index;
        self_[idx] = std::forward<OptionalValueAndIndex>(opt).value().value;
      }
    }
  };

  __host__ __device__
  reference operator[](const grid_executor::index_type&)
  {
    return reference{*this};
  }
};


template<class Container>
__host__ __device__
guarded_container<typename std::decay<Container>::type> make_guarded_container(Container&& value)
{
  return guarded_container<typename std::decay<Container>::type>(std::forward<Container>(value));
}


template<class Factory>
struct guarded_container_factory
{
  mutable Factory factory_;
  size_t user_shape_;

  using container_type = typename std::result_of<Factory(size_t)>::type;

  __agency_hd_warning_disable__
  __host__ __device__
  guarded_container<container_type> operator()(const grid_executor::shape_type& shape) const
  {
    return cuda::detail::make_guarded_container(factory_(user_shape_));
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

    // XXX maybe use whichever of the first two elements of base_executor_type::index_type has larger dimensionality?
    using index_type = size_t;

    template<class T>
    using future = cuda::grid_executor::template future<T>;

    template<class T>
    using container = cuda::detail::array<T>;

    // XXX initialize outer_subscription_ correctly
    __host__ __device__
    flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : outer_subscription_(2),
        base_executor_(base_executor)
    {}

    // XXX needs __host__ __device__
    template<class Function, class Factory1, class T, class Factory2>
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, future<T>& dependency, Factory2 shared_parameter_factory)
    {
      // create a dummy function for partitioning purposes
      auto dummy_function = agency::detail::make_flatten_index_and_invoke<cuda::grid_executor::index_type,T>(f, partition_type{}, shape);

      cuda::detail::guarded_container_factory<Factory1> intermediate_result_factory{result_factory,shape};

      // partition up the iteration space
      auto partitioning = partition(dummy_function, intermediate_result_factory, shape, dependency, shared_parameter_factory, agency::detail::unit_factory());

      // create a function to execute
      auto execute_me = agency::detail::make_flatten_index_and_invoke<cuda::grid_executor::index_type,T>(f, partitioning, shape);

      return base_executor().then_execute(execute_me, intermediate_result_factory, partitioning, dependency, shared_parameter_factory, agency::detail::unit_factory());
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

    template<class Function, class Factory1, class T, class Factory2, class Factory3>
    __host__ __device__
    partition_type partition(const Function& f, const Factory1& result_factory, shape_type shape, const future<T>& dependency, const Factory2& outer_factory, const Factory3& inner_factory) const
    {
      return partition_impl(base_executor().max_shape(f, result_factory, dependency, outer_factory, inner_factory), shape);
    }

    size_t min_inner_size_;
    size_t outer_subscription_;
    base_executor_type base_executor_;
};


} // end agency

