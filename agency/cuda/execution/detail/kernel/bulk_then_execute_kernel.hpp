#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/execution/detail/kernel/launch_kernel.hpp>
#include <agency/cuda/execution/detail/kernel/on_chip_shared_parameter.hpp>
#include <agency/cuda/device.hpp>
#include <agency/cuda/detail/future/async_future.hpp>
#include <memory>

namespace agency
{
namespace cuda
{
namespace detail
{


// XXX consider moving the stuff related to bulk_then_execute_closure into its own header
//     as bulk_then_execution_concurrent_grid.hpp also depends on it
template<size_t block_dimension, class Function, class PredecessorPointer, class ResultPointer, class OuterParameterPointer, class InnerFactory>
struct bulk_then_execute_closure
{
  Function              f_;
  PredecessorPointer    predecessor_ptr_;
  ResultPointer         result_ptr_;
  OuterParameterPointer outer_parameter_ptr_;
  InnerFactory          inner_factory_;

  // this is the implementation for non-void predecessor
  template<class F, class T1, class T2, class T3, class T4>
  __device__ static inline void impl(F f, T1& predecessor, T2& result, T3& outer_param, T4& inner_param)
  {
    f(predecessor, result, outer_param, inner_param);
  }

  // this is the implementation for void predecessor
  template<class F, class T2, class T3, class T4>
  __device__ static inline void impl(F f, agency::detail::unit, T2& result, T3& outer_param, T4& inner_param)
  {
    f(result, outer_param, inner_param);
  }

  template<size_t dimension = block_dimension, __AGENCY_REQUIRES(dimension == 1)>
  __device__ static inline bool is_first_thread_of_block()
  {
#ifdef __CUDA_ARCH__
    return threadIdx.x == 0;
#else
    return false;
#endif
  }

  template<size_t dimension = block_dimension, __AGENCY_REQUIRES(dimension > 1)>
  __device__ static inline bool is_first_thread_of_block()
  {
#ifdef __CUDA_ARCH__
    agency::int3 idx{threadIdx.x, threadIdx.y, threadIdx.z};
#else
    agency::int3 idx{};
#endif

    // XXX this is actually the correct comparison
    //     but this comparison explodes the kernel resource requirements of programs like
    //     testing/unorganized/transpose which launch multidimensional thread blocks.
    //     Those large resource requirements lead to significantly degraded performance.
    //     We should investigate ways to mitigate those requirements and use the correct
    //     comparison. One idea is to teach on_chip_shared_parameter to avoid calling
    //     trivial constructors and destructors.
    //return idx == agency::int3{0,0,0};

    // XXX note that this comparison always fails -- it is incorrect
    return false;
  }

  __device__ inline void operator()()
  {
    // we need to cast each dereference below to convert proxy references to ensure that f() only sees raw references
    // XXX isn't there a more elegant way to deal with this?
    using predecessor_reference = typename std::pointer_traits<PredecessorPointer>::element_type &;
    using result_reference      = typename std::pointer_traits<ResultPointer>::element_type &;
    using outer_param_reference = typename std::pointer_traits<OuterParameterPointer>::element_type &;

    on_chip_shared_parameter<InnerFactory> inner_parameter(is_first_thread_of_block(), inner_factory_);

    impl(
      f_,
      static_cast<predecessor_reference>(*predecessor_ptr_),
      static_cast<result_reference>(*result_ptr_),
      static_cast<outer_param_reference>(*outer_parameter_ptr_),
      inner_parameter.get()
    );
  }
};

template<size_t block_dimension, class Function, class PredecessorPointer, class ResultPointer, class OuterParameterPointer, class InnerFactory>
__host__ __device__
bulk_then_execute_closure<block_dimension,Function,PredecessorPointer,ResultPointer,OuterParameterPointer,InnerFactory>
  make_bulk_then_execute_closure(Function f, PredecessorPointer predecessor_ptr, ResultPointer result_ptr, OuterParameterPointer outer_parameter_ptr, InnerFactory inner_factory)
{
  return bulk_then_execute_closure<block_dimension,Function,PredecessorPointer,ResultPointer,OuterParameterPointer,InnerFactory>{f, predecessor_ptr, result_ptr, outer_parameter_ptr, inner_factory};
}


template<size_t block_dimension, class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
struct bulk_then_execute_kernel
{
  using result_type = agency::detail::result_of_t<ResultFactory()>;
  using outer_arg_type = agency::detail::result_of_t<OuterFactory()>;

  using predecessor_pointer_type = decltype(std::declval<agency::detail::asynchronous_state<T>&>().data());
  using result_pointer_type = decltype(std::declval<agency::detail::asynchronous_state<result_type>&>().data());
  using outer_parameter_pointer_type = decltype(std::declval<agency::detail::asynchronous_state<outer_arg_type>&>().data());

  using closure_type = bulk_then_execute_closure<block_dimension, Function, predecessor_pointer_type, result_pointer_type, outer_parameter_pointer_type, InnerFactory>;

  using type = decltype(&cuda_kernel<closure_type>);

  constexpr static const type value = &cuda_kernel<closure_type>;
};

template<size_t block_dimension, class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
using bulk_then_execute_kernel_t = typename bulk_then_execute_kernel<block_dimension,Function,T,ResultFactory,OuterFactory,InnerFactory>::type;


// this helper function returns a pointer to the kernel launched within launch_bulk_then_execute_kernel_impl()
template<size_t block_dimension, class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
__host__ __device__
bulk_then_execute_kernel_t<block_dimension,Function,T,ResultFactory,OuterFactory,InnerFactory> make_bulk_then_execute_kernel(const Function& f, const asynchronous_state<T>&, const ResultFactory&, const OuterFactory&, const InnerFactory&)
{
  return bulk_then_execute_kernel<block_dimension,Function,T,ResultFactory,OuterFactory,InnerFactory>::value;
}


template<class Shape>
__host__ __device__
::dim3 make_dim3(const Shape& shape)
{
  agency::uint3 temp = agency::detail::shape_cast<uint3>(shape);
  return ::dim3(temp.x, temp.y, temp.z);
}


// this is the main implementation of the other two launch_bulk_then_execute_kernel() functions
template<class Function, class Shape, class T, class ResultFactory, class OuterFactory, class InnerFactory>
__host__ __device__
async_future<agency::detail::result_of_t<ResultFactory()>>
  launch_bulk_then_execute_kernel_impl(device_id device, detail::stream&& stream, Function f, ::dim3 grid_dim, Shape block_dim, const asynchronous_state<T>& predecessor_state, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
{
  // create the asynchronous state to store the result
  using result_type = agency::detail::result_of_t<ResultFactory()>;
  detail::asynchronous_state<result_type> result_state = detail::make_asynchronous_state(result_factory);
  
  // create the asynchronous state to store the outer shared argument
  using outer_arg_type = agency::detail::result_of_t<OuterFactory()>;
  detail::asynchronous_state<outer_arg_type> outer_arg_state = detail::make_asynchronous_state(outer_factory);

  // wrap up f and its arguments into a closure to execute in a kernel
  const size_t block_dimension = agency::detail::shape_size<Shape>::value;
  auto closure = make_bulk_then_execute_closure<block_dimension>(f, predecessor_state.data(), result_state.data(), outer_arg_state.data(), inner_factory);

  // make the kernel to launch
  auto kernel = make_cuda_kernel(closure);

  // launch the kernel
  detail::try_launch_kernel_on_device(kernel, grid_dim, detail::make_dim3(block_dim), 0, stream.native_handle(), device.native_handle(), closure);

  // create the next event
  detail::event next_event(std::move(stream));

  // schedule the outer arg's state for destruction when the next event is complete
  detail::invalidate_and_destroy_when(outer_arg_state, next_event);

  // return a new async_future corresponding to the next event & result state
  return make_async_future(std::move(next_event), std::move(result_state));
}


template<class Function, class Shape, class T, class ResultFactory, class OuterFactory, class InnerFactory>
__host__ __device__
async_future<agency::detail::result_of_t<ResultFactory()>>
launch_bulk_then_execute_kernel(device_id device, Function f, ::dim3 grid_dim, Shape block_dim, async_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
{
  // since we're going to leave the predecessor future valid, we make a new dependent stream before calling launch_bulk_then_execute_kernel_impl()
  return detail::launch_bulk_then_execute_kernel_impl(device, detail::async_future_event(predecessor).make_dependent_stream(device), f, grid_dim, block_dim, detail::async_future_state(predecessor), result_factory, outer_factory, inner_factory);
}


template<class Function, class Shape, class T, class ResultFactory, class OuterFactory, class InnerFactory>
__host__ __device__
async_future<agency::detail::result_of_t<ResultFactory()>>
launch_bulk_then_execute_kernel_and_invalidate_predecessor(device_id device, Function f, ::dim3 grid_dim, Shape block_dim, async_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
{
  // invalidate the future by splitting it into its event and state
  detail::event predecessor_event;
  detail::asynchronous_state<T> predecessor_state;
  agency::tie(predecessor_event, predecessor_state) = detail::invalidate_async_future(predecessor);

  // launch the kernel
  auto result = detail::launch_bulk_then_execute_kernel_impl(device, predecessor_event.make_dependent_stream_and_invalidate(device), f, grid_dim, block_dim, predecessor_state, result_factory, outer_factory, inner_factory);

  // schedule the predecessor's state for destruction when the result future's event is complete
  detail::invalidate_and_destroy_when(predecessor_state, detail::async_future_event(result));

  return result;
}


template<size_t block_dimension, class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
__host__ __device__
int max_block_size_of_bulk_then_execute_kernel(const device_id& device, const Function& f, const async_future<T>& predecessor, const ResultFactory& result_factory, const OuterFactory& outer_factory, const InnerFactory& inner_factory)
{
  // temporarily switch the CUDA runtime's current device to the given device
  scoped_device scope(device);

  // get a pointer to the kernel launched by launch_bulk_then_execute_kernel()
  constexpr auto kernel = bulk_then_execute_kernel<block_dimension,Function,T,ResultFactory,OuterFactory,InnerFactory>::value;

  // get the kernel's attributes
  cudaFuncAttributes attr;
  detail::throw_on_error(cudaFuncGetAttributes(&attr, kernel), "cuda::detail::max_block_size_of_bulk_then_execute_grid(): CUDA error after cudaFuncGetAttributes()");

  // return the attribute of interest
  return attr.maxThreadsPerBlock;
}


} // end detail
} // end cuda
} // end agency

