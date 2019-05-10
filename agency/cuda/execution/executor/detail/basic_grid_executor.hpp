#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/coordinate/detail/shape/shape_size.hpp>
#include <agency/tuple.hpp>
#include <agency/cuda/execution/detail/kernel/bulk_then_execute_concurrent_kernel.hpp>
#include <agency/cuda/execution/detail/kernel/bulk_then_execute_kernel.hpp>
#include <agency/cuda/memory/allocator/allocator.hpp>
#include <agency/cuda/detail/future/async_future.hpp>
#include <agency/cuda/detail/future/shared_future.hpp>
#include <agency/cuda/device.hpp>
#include <agency/detail/scoped_in_place_type.hpp>
#include <agency/cuda/detail/concurrency/block_barrier.hpp>
#include <agency/cuda/detail/concurrency/grid_barrier.hpp>
#include <tuple>


namespace agency
{
namespace cuda
{
namespace detail
{


// these functions create an agent's outer_index_type & inner_index_type from blockIdx & threadIdx, respectively
// use of CUDA built-in variables blockIdx and threadIdx is guarded by __CUDA_ARCH__ to allow non-CUDA compilers to
// parse this header file

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           // XXX we should use index_size, but there's no such trait at the moment
           agency::detail::shape_size<I>::value == 1
         )>
inline __device__
Index make_outer_index()
{
#ifdef __CUDA_ARCH__
  return Index{blockIdx.x};
#else
  return Index{};
#endif
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 2
         )>
inline __device__
Index make_outer_index()
{
#ifdef __CUDA_ARCH__
  return Index{blockIdx.x, blockIdx.y};
#else
  return Index{};
#endif 
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 3
         )>
inline __device__
Index make_outer_index()
{
#ifdef __CUDA_ARCH__
  return Index{blockIdx.x, blockIdx.y, blockIdx.z};
#else
  return Index{};
#endif
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 1
         )>
inline __device__
Index make_inner_index()
{
#ifdef __CUDA_ARCH__
  return Index{threadIdx.x};
#else
  return Index{};
#endif
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 2
         )>
inline __device__
Index make_inner_index()
{
#ifdef __CUDA_ARCH__
  return Index{threadIdx.x, threadIdx.y};
#else
  return Index{};
#endif
}

template<class Index,
         class I = Index,
         __AGENCY_REQUIRES(
           agency::detail::shape_size<I>::value == 3
         )>
inline __device__
Index make_inner_index()
{
#ifdef __CUDA_ARCH__
  return Index{threadIdx.x, threadIdx.y, threadIdx.z};
#else
  return Index{};
#endif
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


template<class OuterBulkGuarantee, class Shape, class Index = Shape>
class basic_grid_executor
{
  static_assert(std::is_same<bulk_guarantee_t::parallel_t, OuterBulkGuarantee>::value or std::is_same<bulk_guarantee_t::concurrent_t, OuterBulkGuarantee>::value,
                "basic_grid_executor's OuterBulkGuarantee type must be bulk_guarantee_t::parallel_t or bulk_guarantee_t::concurrent_t.");

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
    using shape_type = Shape;

    using index_type = Index;

    template<class T>
    using future = cuda::async_future<T>;

    template<class T>
    using allocator = cuda::allocator<T>;

    using barrier_type = agency::detail::scoped_in_place_type_t<
      typename std::conditional<
        std::is_same<bulk_guarantee_t::parallel_t, OuterBulkGuarantee>::value,
        void,                // parallel grid_executors have no outer barrier type
        detail::grid_barrier // concurrent grid_executors' outer barrier_type is grid_barrier
      >::type,
      detail::block_barrier  // the inner barrier_type is always block_barrier
    >;

    __host__ __device__
    constexpr explicit basic_grid_executor(device_id device = device_id(0))
      : device_(device)
    {}


    __host__ __device__
    constexpr static bulk_guarantee_t::scoped_t<
      OuterBulkGuarantee,
      bulk_guarantee_t::concurrent_t
    > query(const bulk_guarantee_t&)
    {
      return bulk_guarantee_t::scoped(
        // CUDA thread blocks may be parallel or concurrent with respect to each other
        OuterBulkGuarantee(),

        // CUDA threads are always concurrent within a thread block
        bulk_guarantee_t::concurrent_t()
      );
    }


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


    __host__ __device__
    friend bool operator==(const basic_grid_executor& a, const basic_grid_executor& b) noexcept
    {
      return a.device() == b.device();
    }


    __host__ __device__
    friend bool operator!=(const basic_grid_executor& a, const basic_grid_executor& b) noexcept
    {
      return !(a == b);
    }


  private:
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    static async_future<agency::detail::result_of_t<ResultFactory()>>
      launch_kernel_and_invalidate_predecessor(bulk_guarantee_t::parallel_t, 
                                               device_id device,
                                               Function f,
                                               dim3 grid_dim,
                                               inner_shape_type block_dim,
                                               async_future<T>& predecessor,
                                               ResultFactory result_factory,
                                               OuterFactory outer_factory,
                                               InnerFactory inner_factory)
    {
      return detail::launch_bulk_then_execute_kernel_and_invalidate_predecessor(device, f, grid_dim, block_dim, predecessor, result_factory, outer_factory, inner_factory);
    }


    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    static async_future<agency::detail::result_of_t<ResultFactory()>>
      launch_kernel_and_invalidate_predecessor(bulk_guarantee_t::concurrent_t, 
                                               device_id device,
                                               Function f,
                                               dim3 grid_dim,
                                               inner_shape_type block_dim,
                                               async_future<T>& predecessor,
                                               ResultFactory result_factory,
                                               OuterFactory outer_factory,
                                               InnerFactory inner_factory)
    {
      return detail::launch_bulk_then_execute_concurrent_kernel_and_invalidate_predecessor(device, f, grid_dim, block_dim, predecessor, result_factory, outer_factory, inner_factory);
    }


  public:
    // this overload of bulk_then_execute() receives a future<T> predecessor and invalidates it
    __agency_exec_check_disable__
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
      
      return launch_kernel_and_invalidate_predecessor(OuterBulkGuarantee(), device(), closure, grid_dim, agency::get<1>(shape), predecessor, result_factory, outer_factory, inner_factory);
    }


  private:
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    static async_future<agency::detail::result_of_t<ResultFactory()>>
      launch_kernel_and_leave_predecessor_valid(bulk_guarantee_t::parallel_t, 
                                                device_id device,
                                                Function f,
                                                dim3 grid_dim,
                                                inner_shape_type block_dim,
                                                async_future<T>& predecessor,
                                                ResultFactory result_factory,
                                                OuterFactory outer_factory,
                                                InnerFactory inner_factory)
    {
      return detail::launch_bulk_then_execute_kernel(device, f, grid_dim, block_dim, predecessor, result_factory, outer_factory, inner_factory);
    }


    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    static async_future<agency::detail::result_of_t<ResultFactory()>>
      launch_kernel_and_leave_predecessor_valid(bulk_guarantee_t::concurrent_t, 
                                                device_id device,
                                                Function f,
                                                dim3 grid_dim,
                                                inner_shape_type block_dim,
                                                async_future<T>& predecessor,
                                                ResultFactory result_factory,
                                                OuterFactory outer_factory,
                                                InnerFactory inner_factory)
    {
      return detail::launch_bulk_then_execute_concurrent_kernel(device, f, grid_dim, block_dim, predecessor, result_factory, outer_factory, inner_factory);
    }


  public:
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
      return launch_kernel_and_leave_predecessor_valid(OuterBulkGuarantee(), device(), closure, grid_dim, agency::get<1>(shape), async_predecessor, result_factory, outer_factory, inner_factory);
    }


    // this overload of bulk_then_execute() consumes a generic Future predecessor
    // it is unimplemented and throws an exception at runtime
    template<class Function, class Future, class ResultFactory, class OuterFactory, class InnerFactory>
    async_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory) const
    {
      throw std::runtime_error("basic_grid_executor::bulk_then_execute() with generic Future predecessor: unimplemented.");
    }


    // this function maximizes each of shape's non-zero elements
    // XXX probably needs a better name
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    shape_type max_shape_dimensions(Function f, shape_type shape, async_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory) const
    {
      unsigned int outer_size = agency::get<0>(shape);
      unsigned int inner_size = agency::get<1>(shape);

      if(inner_size == 0)
      {
        inner_size = max_inner_size(f, predecessor, result_factory, outer_factory, inner_factory);
      }

      if(outer_size == 0)
      {
        outer_size = max_outer_size(f, inner_size, predecessor, result_factory, outer_factory, inner_factory);
      }

      return shape_type{outer_size, inner_size};
    }


  private:
    // returns the largest possible inner group size for the given function to execute
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory,
             class BulkGuarantee = OuterBulkGuarantee,
             __AGENCY_REQUIRES(
               std::is_same<BulkGuarantee, bulk_guarantee_t::parallel_t>::value
             )>
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


    // returns the largest possible inner group size for the given function to execute
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory,
             class BulkGuarantee = OuterBulkGuarantee,
             __AGENCY_REQUIRES(
               std::is_same<BulkGuarantee, bulk_guarantee_t::concurrent_t>::value
             )>
    __host__ __device__
    std::size_t max_inner_size(Function f,
                               async_future<T>& predecessor,
                               ResultFactory result_factory,
                               OuterFactory outer_factory,
                               InnerFactory inner_factory) const
    {
      constexpr size_t block_dimension = agency::detail::shape_size<inner_shape_type>::value;
      int max_block_size = detail::max_block_size_of_bulk_then_execute_concurrent_kernel<block_dimension>(device(), invoke_with_agent_index<Function,index_type>{f}, predecessor, result_factory, outer_factory, inner_factory);
      return static_cast<std::size_t>(max_block_size);
    }

    // returns the largest possible parallel outer group size for the given inner group size and function to execute
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory,
             class BulkGuarantee = OuterBulkGuarantee,
             __AGENCY_REQUIRES(
               std::is_same<BulkGuarantee, bulk_guarantee_t::parallel_t>::value
             )>
    __host__ __device__
    std::size_t max_outer_size(Function f,
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
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory,
             class BulkGuarantee = OuterBulkGuarantee,
             __AGENCY_REQUIRES(
               std::is_same<BulkGuarantee, bulk_guarantee_t::concurrent_t>::value
             )>
    __host__ __device__
    std::size_t max_outer_size(Function f,
                               std::size_t inner_size,
                               async_future<T>& predecessor,
                               ResultFactory result_factory,
                               OuterFactory outer_factory,
                               InnerFactory inner_factory) const
    {
      invoke_with_agent_index<Function,index_type> closure{f};
      return detail::max_grid_size_of_bulk_then_execute_concurrent_kernel(device(), closure, inner_size, predecessor, result_factory, outer_factory, inner_factory);
    }


    // this function extracts the outer_shape from the executor's shape_type
    // and converts it into a dim3 which describes the gridDim of the CUDA
    // kernel to launch
    // there is not an analogous make_block_dim() function because that conversion
    // happens within the implementation of the lower-level kernel launch interface.
    __host__ __device__
    static dim3 make_grid_dim(shape_type shape)
    {
      // cast the 0th element of shape into a uint3
      uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::get<0>(shape));

      // unpack outer_shape into dim3
      return dim3(outer_shape.x, outer_shape.y, outer_shape.z);
    }


    device_id device_;
};


} // end detail
} // end cuda
} // end agency

