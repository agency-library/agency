#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/cuda/detail/tuple.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/device.hpp>
#include <agency/cuda/detail/bind.hpp>
#include <agency/cuda/detail/memory/unique_ptr.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/cuda/detail/when_all_execute_and_select.hpp>
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

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/detail/minmax.h>

#include <memory>
#include <iostream>
#include <exception>
#include <cstring>
#include <type_traits>
#include <cassert>


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
    using future = cuda::async_future<T>;


    template<class T>
    using container = detail::array<T, shape_type>;


    __host__ __device__
    explicit basic_grid_executor(device_id device = detail::current_device())
      : device_(device)
    {}


    __host__ __device__
    device_id device() const
    {
      return device_;
    }

    
    __host__ __device__
    void device(device_id device)
    {
      device_ = device;
    }

    
    __host__ __device__
    async_future<void> make_ready_future()
    {
      return cuda::make_ready_async_future();
    }

    template<size_t... Indices, class Function, class TupleOfFutures, class Factory1, class Factory2>
    __host__ __device__
    async_future<detail::when_all_execute_and_select_result_t<agency::detail::index_sequence<Indices...>, agency::detail::decay_t<TupleOfFutures>>>
      when_all_execute_and_select(Function f,
                                  shape_type shape,
                                  TupleOfFutures&& tuple_of_futures,
                                  Factory1 outer_factory,
                                  Factory2 inner_factory)
    {
      return detail::when_all_execute_and_select<Indices...>(f, shape, ThisIndexFunction(), std::forward<TupleOfFutures>(tuple_of_futures), outer_factory, inner_factory, device());
    }

    template<class Function, class Factory1, class T, class Factory2, class Factory3,
             class = agency::detail::result_of_continuation_t<
               Function,
               index_type,
               async_future<T>,
               agency::detail::result_of_factory_t<Factory2>&,
               agency::detail::result_of_factory_t<Factory3>&
             >
            >
    __host__ __device__
    async_future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, async_future<T>& fut, Factory2 outer_factory, Factory3 inner_factory)
    {
      return fut.bulk_then(f, result_factory, shape, ThisIndexFunction(), outer_factory, inner_factory, device());
    }


    template<class Function, class Factory1, class T, class Factory2, class Factory3,
             class = agency::detail::result_of_continuation_t<
               Function,
               index_type,
               shared_future<T>,
               agency::detail::result_of_factory_t<Factory2>&,
               agency::detail::result_of_factory_t<Factory3>&
             >
            >
    async_future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, shared_future<T>& fut, Factory2 outer_factory, Factory3 inner_factory)
    {
      using result_type = async_future<typename std::result_of<Factory1(shape_type)>::type>;
      auto intermediate_future = fut.bulk_then(f, result_factory, shape, ThisIndexFunction(), outer_factory, inner_factory, device());
      return std::move(intermediate_future.get<result_type>());
    }


    // this is exposed because it's necessary if a client wants to compute occupancy
    // alternatively, cuda_executor could report occupancy of a Function for a given block size
    // XXX probably shouldn't expose this -- max_shape() seems good enough

    template<class Function, class Factory1, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    void* then_execute_kernel(const Function& f, const Factory1& result_factory, const async_future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory) const
    {
      return fut.bulk_then_kernel(f, result_factory, shape_type{}, ThisIndexFunction(), outer_factory, inner_factory, device());
    }

    template<class Container, class Function, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    void* then_execute_kernel(const Function& f, const async_future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory) const
    {
      agency::detail::factory<Container> result_factory;
      return then_execute_kernel(f, result_factory, fut, outer_factory, inner_factory);
    }

    template<class Function, class T, class OuterFactory, class InnerFactory>
    __host__ __device__
    void* then_execute_kernel(const Function& f, const async_future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory) const
    {
      using container_type = agency::detail::new_executor_traits_detail::discarding_container;
      auto g = agency::detail::invoke_and_return_unit<Function>{f};
      return then_execute_kernel<container_type>(g, fut, outer_factory, inner_factory);
    }

    template<class Function, class T>
    __host__ __device__
    static void* then_execute_kernel(const Function& f, const async_future<T>& fut)
    {
      // XXX if T is void, then we only want to take the first parameter and invoke, because there is no second parameter
      //     so what we really want is something like take_at_most_first_two_parameters_and_invoke<Function>
      auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};
      return then_execute_kernel(g, fut, agency::detail::unit_factory(), agency::detail::unit_factory());
    }


    template<class Function>
    __host__ __device__
    static void* then_execute_kernel(const Function& f)
    {
      return then_execute_kernel(f, async_future<void>());
    }


    // XXX we should eliminate these functions below since executor_traits implements them for us

    template<class Function>
    __host__ __device__
    async_future<void> async_execute(Function f, shape_type shape)
    {
      auto ready = make_ready_future();
      return agency::executor_traits<basic_grid_executor>::then_execute(*this, f, shape, ready);
    }
    

    template<class Function, class Factory1, class Factory2>
    __host__ __device__
    async_future<void> async_execute(Function f, shape_type shape, Factory1 outer_factory, Factory2 inner_factory)
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
    device_id device_;
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
      if(current_device != device().native_handle())
      {
#  ifndef __CUDA_ARCH__
        detail::throw_on_error(cudaSetDevice(device().native_handle()), "cuda::grid_executor::max_shape(): cudaSetDevice()");
#  else
        detail::throw_on_error(cudaErrorNotSupported, "cuda::grid_executor::max_shape(): cudaSetDevice only allowed in __host__ code");
#  endif // __CUDA_ARCH__
      }

      size_t max_grid_dimension_x = detail::maximum_grid_size_x(device());

      cudaFuncAttributes attr{};
      detail::throw_on_error(cudaFuncGetAttributes(&attr, fun_ptr),
                             "cuda::grid_executor::max_shape(): cudaFuncGetAttributes");

      // restore current device
      if(current_device != device().native_handle())
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
    shape_type max_shape(const Function& f, const async_future<T>& fut)
    {
      return max_shape_impl(then_execute_kernel(f,fut));
    }

    template<class T, class Function, class Factory1, class Factory2>
    __host__ __device__
    shape_type max_shape(const Function& f, const async_future<T>& fut, const Factory1& outer_factory, const Factory2& inner_factory) const
    {
      return max_shape_impl(then_execute_kernel(f, fut, outer_factory, inner_factory));
    }

    template<class Function, class Factory1, class T, class Factory2, class Factory3>
    __host__ __device__
    shape_type max_shape(const Function& f, const Factory1& result_factory, const async_future<T>& fut, const Factory2& outer_factory, const Factory3& inner_factory) const
    {
      return max_shape_impl(then_execute_kernel(f, result_factory, fut, outer_factory, inner_factory));
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


} // end cuda
} // end agency

