#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/coordinate.hpp>
#include <agency/cuda/device.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/memory/allocator.hpp>
#include <agency/cuda/execution/detail/kernel/bulk_then_execute_kernel.hpp>
#include <tuple>


namespace agency
{
namespace cuda
{
namespace detail
{


// these functions create an agent's outer_index_type & inner_index_type from blockIdx & threadIdx, respectively

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           // XXX we should use index_size, but there's no such trait at the moment
           agency::detail::shape_size<I>::value == 1
         )>
inline __device__
Index make_outer_index()
{
  return Index{blockIdx.x};
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 2
         )>
inline __device__
Index make_outer_index()
{
  return Index{blockIdx.x, blockIdx.y};
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 3
         )>
inline __device__
Index make_outer_index()
{
  return Index{blockIdx.x, blockIdx.y, blockIdx.z};
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 1
         )>
inline __device__
Index make_inner_index()
{
  return Index{threadIdx.x};
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 2
         )>
inline __device__
Index make_inner_index()
{
  return Index{threadIdx.x, threadIdx.y};
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 3
         )>
inline __device__
Index make_inner_index()
{
  return Index{threadIdx.x, threadIdx.y, threadIdx.z};
}

// this function uses make_outer_index & make_inner_index to make an agent's index_type
template<class Index>
inline __device__
Index make_index()
{
  using outer_index_type = typename std::tuple_element<0,Index>::type;
  using inner_index_type = typename std::tuple_element<1,Index>::type;

  return Index{make_outer_index<outer_index_type>(), make_inner_index<inner_index_type>()};
}


template<class Index>
struct make_index_functor
{
  __device__
  Index operator()() const
  {
    return make_index<Index>();
  }
};


template<class Function, class Index>
struct invoke_with_agent_index
{
  mutable Function f_;

  template<class... Args>
  inline __device__
  void operator()(Args&&... args) const
  {
    // call f with args but insert the agent's index as the first parameter
    f_(make_index<Index>(), std::forward<Args>(args)...);
  }
};


template<class Shape, class Index = Shape>
class basic_grid_executor
{
  using outer_shape_type = typename std::tuple_element<0,Shape>::type;
  using inner_shape_type = typename std::tuple_element<1,Shape>::type;

  using outer_index_type = typename std::tuple_element<0,Index>::type;
  using inner_index_type = typename std::tuple_element<1,Index>::type;

  static_assert(agency::detail::shape_size<Shape>::value == 2,
                "basic_grid_executor's Shape type must have two elements.");

  static_assert(agency::detail::shape_size<outer_shape_type>::value <= 3,
                "The first element of basic_grid_executor's Shape type must have three or less elements.");

  static_assert(agency::detail::shape_size<inner_shape_type>::value <= 3,
                "The second element of basic_grid_executor's Shape type must have three or less elements.");

  static_assert(agency::detail::shape_size<Index>::value == agency::detail::shape_size<Shape>::value,
                "basic_grid_executor's Index type must have as many elements as its Shape type.");

  static_assert(agency::detail::shape_size<outer_index_type>::value == agency::detail::shape_size<outer_shape_type>::value,
                "The dimensions of basic_grid_executor's Index type must match the dimensions of its Shape type.");

  static_assert(agency::detail::shape_size<inner_index_type>::value == agency::detail::shape_size<inner_shape_type>::value,
                "The dimensions of basic_grid_executor's Index type must match the dimensions of its Shape type.");


  public:
    // XXX generalize to OuterExecutionCategory?
    using execution_category =
      scoped_execution_tag<
        parallel_execution_tag,
        concurrent_execution_tag
      >;

    using shape_type = Shape;

    using index_type = Index;

    template<class T>
    using future = cuda::async_future<T>;

    template<class T>
    using allocator = cuda::allocator<T>;

    __host__ __device__
    explicit basic_grid_executor(device_id device = device_id(0))
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
    async_future<void> make_ready_future() const
    {
      return cuda::make_ready_async_future();
    }


    // this overload of bulk_then_execute() receives a future<T> predecessor and invalidates it
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    async_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f,
                        shape_type shape,
                        async_future<T>& predecessor,
                        ResultFactory result_factory,
                        OuterFactory outer_factory,
                        InnerFactory inner_factory) const
    {
      // decompose shape into the kernel launch configuration
      dim3 grid_dim  = this->make_grid_dim(shape);

      // create a closure wrapping f which will pass f the execution agent's index as its first parameter
      invoke_with_agent_index<Function,index_type> closure{f};
      
      return detail::launch_bulk_then_execute_kernel_and_invalidate_predecessor(device(), closure, grid_dim, agency::detail::get<1>(shape), predecessor, result_factory, outer_factory, inner_factory);
    }


    // this overload of bulk_then_execute() receives a shared_future<T> predecessor and leaves it valid
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    async_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f,
                        shape_type shape,
                        shared_future<T>& predecessor,
                        ResultFactory result_factory,
                        OuterFactory outer_factory,
                        InnerFactory inner_factory) const
    {
      // decompose shape into the kernel launch configuration
      dim3 grid_dim  = this->make_grid_dim(shape);

      // create a closure wrapping f which will pass f the execution agent's index as its first parameter
      invoke_with_agent_index<Function,index_type> closure{f};

      // get access to the shared_future's underlying future
      agency::cuda::future<T>& underlying_predecessor = detail::underlying_future(predecessor);

      // cast the underlying_predecessor to an async_future
      // XXX we need to handle cases where the predecessor is not an async_future
      async_future<T>& async_predecessor = underlying_predecessor.template get<async_future<T>>();

      // implement with lower-level kernel launch functionality
      using result_type = agency::detail::result_of_t<ResultFactory()>;
      return detail::launch_bulk_then_execute_kernel(device(), closure, grid_dim, agency::detail::get<1>(shape), async_predecessor, result_factory, outer_factory, inner_factory);
    }


    // this overload of bulk_then_execute() consumes a generic Future predecessor
    // it is unimplemented and throws an exception at runtime
    template<class Function, class Future, class ResultFactory, class OuterFactory, class InnerFactory>
    async_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory) const
    {
      throw std::runtime_error("basic_grid_executor::bulk_then_execute() with generic Future predecessor: unimplemented.");
    }


  protected:
    // returns the largest possible inner group size for the given function to execute
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    std::size_t max_inner_size(Function f,
                               async_future<T>& predecessor,
                               ResultFactory result_factory,
                               OuterFactory outer_factory,
                               InnerFactory inner_factory) const
    {
      constexpr size_t block_dimension = agency::detail::shape_size<inner_shape_type>::value;
      int max_block_size = detail::max_block_size_of_bulk_then_execute_kernel<block_dimension>(device(), invoke_with_agent_index<Function,index_type>{f}, predecessor, result_factory, outer_factory, inner_factory);
      return static_cast<std::size_t>(max_block_size);
    }

    // returns the largest possible parallel outer group size for the given inner group size and function to execute
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    std::size_t max_parallel_outer_size(Function f,
                                        std::size_t inner_size,
                                        async_future<T>& predecessor,
                                        ResultFactory result_factory,
                                        OuterFactory outer_factory,
                                        InnerFactory inner_factory) const
    {
      int max_grid_size = detail::maximum_grid_size_x(device());
      return static_cast<std::size_t>(max_grid_size);
    }

    // returns the largest possible concurrent outer group size for the given inner group size and function to execute
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    std::size_t max_concurrent_outer_size(Function f,
                                          std::size_t inner_size,
                                          async_future<T>& predecessor,
                                          ResultFactory result_factory,
                                          OuterFactory outer_factory,
                                          InnerFactory inner_factory) const
    {
      printf("basic_grid_executor::max_concurrent_outer_size(): unimplemented.\n");
      assert(0);
    }

  private:
    // this function extracts the outer_shape from the executor's shape_type
    // and converts it into a dim3 which describes the gridDim of the CUDA
    // kernel to launch
    // there is not an analogous make_block_dim() function because that conversion
    // happens within the implementation of the lower-level kernel launch interface.
    __host__ __device__
    static dim3 make_grid_dim(shape_type shape)
    {
      // cast the 0th element of shape into a uint3
      uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));

      // unpack outer_shape into dim3
      return dim3(outer_shape.x, outer_shape.y, outer_shape.z);
    }

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

    // XXX does any part of Agency actually use this function? maybe we should just get rid of it
    //     in favor of something that works more like the overload below
    __host__ __device__
    shape_type max_shape_dimensions() const
    {
      // XXX it's not clear that this is correct
      return shape_type{detail::maximum_grid_size_x(device()), 256};
    }

    // this function maximizes each of shape's non-zero elements
    // XXX probably needs a better name
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    shape_type max_shape_dimensions(Function f, shape_type shape, async_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory) const
    {
      unsigned int outer_size = agency::detail::get<0>(shape);
      unsigned int inner_size = agency::detail::get<1>(shape);

      if(inner_size == 0)
      {
        inner_size = max_inner_size(f, predecessor, result_factory, outer_factory, inner_factory);
      }

      if(outer_size == 0)
      {
        outer_size = max_parallel_outer_size(f, inner_size, predecessor, result_factory, outer_factory, inner_factory);
      }

      return shape_type{outer_size, inner_size};
    }
};


class grid_executor_2d : public detail::basic_grid_executor<point<agency::uint2,2>>
{
  public:
    using detail::basic_grid_executor<point<agency::uint2,2>>::basic_grid_executor;

    // XXX implement unit_shape()

    // XXX implement max_shape_dimensions()
};


} // end cuda
} // end agency

