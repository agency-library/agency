#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/cuda/detail/tuple.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/device.hpp>
#include <agency/cuda/memory/detail/unique_ptr.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/coordinate.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/factory.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/detail/array.hpp>
#include <agency/detail/type_traits.hpp>

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


// this_cuda_thread_index maps the builtin threadIdx to the given Index type
// Index must have 1, 2, or 3 elements

template<class Index>
__device__
typename std::enable_if<
  agency::detail::index_size<Index>::value == 1,
  Index
>::type
  this_cuda_thread_index()
{
  return Index{threadIdx.x};
}

template<class Index>
__device__
typename std::enable_if<
  agency::detail::index_size<Index>::value == 2,
  Index
>::type
  this_cuda_thread_index()
{
  return Index{threadIdx.x, threadIdx.y};
}

template<class Index>
__device__
typename std::enable_if<
  agency::detail::index_size<Index>::value == 3,
  Index
>::type
  this_cuda_thread_index()
{
  return Index{threadIdx.x, threadIdx.y, threadIdx.z};
}


// this_cuda_block_index maps the builtin blockIdx to the given Index type
// Index must have 1, 2, or 3 elements

template<class Index>
__device__
typename std::enable_if<
  agency::detail::index_size<Index>::value == 1,
  Index
>::type
  this_cuda_block_index()
{
  return Index{blockIdx.x};
}

template<class Index>
__device__
typename std::enable_if<
  agency::detail::index_size<Index>::value == 2,
  Index
>::type
  this_cuda_block_index()
{
  return Index{blockIdx.x, blockIdx.y};
}

template<class Index>
__device__
typename std::enable_if<
  agency::detail::index_size<Index>::value == 3,
  Index
>::type
  this_cuda_block_index()
{
  return Index{blockIdx.x, blockIdx.y, blockIdx.z};
}


// this_cuda_index() maps the builtins blockIdx & threadIdx to the given Index type
// Index must have 2 elements
template<class Index,
         class = typename std::enable_if<
           agency::detail::index_size<Index>::value == 2
         >::type>
__device__
Index this_cuda_index()
{
  using outer_index_type = typename std::tuple_element<0,Index>::type;
  using inner_index_type = typename std::tuple_element<1,Index>::type;

  return Index{
    cuda::detail::this_cuda_block_index<outer_index_type>(),
    cuda::detail::this_cuda_thread_index<inner_index_type>()
  };
}


// this is a functor wrapping this_cuda_index()
template<class Index>
struct this_cuda_index_functor
{
  __device__
  Index operator()() const
  {
    return cuda::detail::this_cuda_index<Index>();
  }
};


template<class Shape, class Index = Shape>
class basic_grid_executor
{
  static_assert(std::tuple_size<Shape>::value == std::tuple_size<Index>::value,
                "basic_grid_executor's Shape and Index types must have the same number of elements.");

  public:
    using execution_category =
      scoped_execution_tag<
        parallel_execution_tag,
        concurrent_execution_tag
      >;


    using shape_type = Shape;


    using index_type = Index;


  private:
    // this is a functor that will map the CUDA builtin variables blockIdx & threadIdx to an Index
    using this_index_function_type = this_cuda_index_functor<index_type>;


  public:
    template<class T>
    using future = cuda::async_future<T>;


    template<class T>
    using allocator = cuda::allocator<T>;


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


    // this overload of bulk_then_execute() consumes a future<T> predecessor
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    async_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
    {
      return predecessor.bulk_then(f, shape, this_index_function_type(), result_factory, outer_factory, inner_factory, device());
    }


    // this overload of bulk_then_execute() consumes a shared_future<T> predecessor
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    async_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, shared_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
    {
      using result_future_type = async_future<agency::detail::result_of_t<ResultFactory()>>;

      auto intermediate_future = predecessor.bulk_then(f, shape, this_index_function_type(), result_factory, outer_factory, inner_factory, device());

      // convert the intermediate future into the type of future we need to return
      return std::move(intermediate_future.template get<result_future_type>());
    }


  private:
    device_id device_;
};


} // end detail


class grid_executor : public detail::basic_grid_executor<agency::uint2>
{
  public:
    using detail::basic_grid_executor<agency::uint2>::basic_grid_executor;

    __host__ __device__
    shape_type unit_shape() const
    {
      return shape_type{detail::number_of_multiprocessors(device()), 256};
    }

    __host__ __device__
    shape_type max_shape_dimensions() const
    {
      return shape_type{detail::maximum_grid_size_x(device()), 256};
    }
};


class grid_executor_2d : public detail::basic_grid_executor<point<agency::uint2,2>>
{
  public:
    using detail::basic_grid_executor<point<agency::uint2,2>>::basic_grid_executor;

    // XXX implement max_shape_dimensions()
};


} // end cuda
} // end agency

