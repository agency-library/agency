#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/execution/detail/kernel/launch_cooperative_kernel.hpp>
#include <agency/cuda/execution/detail/kernel/bulk_then_execute_kernel.hpp>
#include <agency/cuda/detail/future/async_future.hpp>
#include <agency/cuda/device.hpp>
#include <agency/coordinate/detail/shape/shape_size.hpp>
#include <agency/coordinate/detail/shape/shape_cast.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/tuple.hpp>
#include <utility>


namespace agency
{
namespace cuda
{
namespace detail
{


// this is the main implementation of the other two launch_bulk_then_execute_concurrent_kernel() functions
template<class Function, class Shape, class T, class ResultFactory, class OuterFactory, class InnerFactory>
agency::cuda::async_future<agency::detail::result_of_t<ResultFactory()>>
  launch_bulk_then_execute_concurrent_kernel_impl(agency::cuda::device_id device,
                                                  agency::cuda::detail::stream&& stream, Function f,
                                                  ::dim3 grid_dim,
                                                  Shape block_dim,
                                                  const agency::cuda::detail::asynchronous_state<T>& predecessor_state,
                                                  ResultFactory result_factory,
                                                  OuterFactory outer_factory,
                                                  InnerFactory inner_factory)
{
  // create the asynchronous state to store the result
  using result_type = agency::detail::result_of_t<ResultFactory()>;
  detail::asynchronous_state<result_type> result_state = detail::make_asynchronous_state(result_factory);
  
  // create the asynchronous state to store the outer shared argument
  using outer_arg_type = agency::detail::result_of_t<OuterFactory()>;
  detail::asynchronous_state<outer_arg_type> outer_arg_state = detail::make_asynchronous_state(outer_factory);

  // wrap up f and its arguments into a closure to execute in a kernel
  const size_t block_dimension = agency::detail::shape_size<Shape>::value;
  auto closure = detail::make_bulk_then_execute_closure<block_dimension>(f, predecessor_state.data(), result_state.data(), outer_arg_state.data(), inner_factory);

  // make the kernel to launch
  auto kernel = detail::make_cuda_kernel(closure);

  // launch the kernel
  detail::try_launch_cooperative_kernel_on_device(kernel, grid_dim, detail::make_dim3(block_dim), 0, stream.native_handle(), device.native_handle(), closure);

  // create the next event
  detail::event next_event(std::move(stream));

  // schedule the outer arg's state for destruction when the next event is complete
  detail::invalidate_and_destroy_when(outer_arg_state, next_event);

  // return a new async_future corresponding to the next event & result state
  return detail::make_async_future(std::move(next_event), std::move(result_state));
}


template<class Function, class Shape, class T, class ResultFactory, class OuterFactory, class InnerFactory>
agency::cuda::async_future<agency::detail::result_of_t<ResultFactory()>>
launch_bulk_then_execute_concurrent_kernel(agency::cuda::device_id device, Function f, ::dim3 grid_dim, Shape block_dim, agency::cuda::async_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
{
  // since we're going to leave the predecessor future valid, we make a new dependent stream before calling launch_bulk_then_execute_kernel_impl()
  return detail::launch_bulk_then_execute_concurrent_kernel_impl(device, detail::async_future_event(predecessor).make_dependent_stream(device), f, grid_dim, block_dim, detail::async_future_state(predecessor), result_factory, outer_factory, inner_factory);
}


template<class Function, class Shape, class T, class ResultFactory, class OuterFactory, class InnerFactory>
agency::cuda::async_future<agency::detail::result_of_t<ResultFactory()>>
launch_bulk_then_execute_concurrent_kernel_and_invalidate_predecessor(agency::cuda::device_id device, Function f, ::dim3 grid_dim, Shape block_dim, agency::cuda::async_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
{
  // invalidate the future by splitting it into its event and state
  detail::event predecessor_event;
  detail::asynchronous_state<T> predecessor_state;
  agency::tie(predecessor_event, predecessor_state) = detail::invalidate_async_future(predecessor);

  // execute the grid
  auto result = detail::launch_bulk_then_execute_concurrent_kernel_impl(device, predecessor_event.make_dependent_stream_and_invalidate(device), f, grid_dim, block_dim, predecessor_state, result_factory, outer_factory, inner_factory);

  // schedule the predecessor's state for destruction when the result future's event is complete
  detail::invalidate_and_destroy_when(predecessor_state, detail::async_future_event(result));

  return result;
}


template<size_t block_dimension, class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
__host__ __device__
int max_block_size_of_bulk_then_execute_concurrent_kernel(const agency::cuda::device_id& device, const Function&, const agency::cuda::async_future<T>&, const ResultFactory&, const OuterFactory&, const InnerFactory&)
{
  // temporarily switch the CUDA runtime's current device to the given device
  scoped_device scope(device);

  // get a pointer to the kernel which bulk_then_execute_concurent_grid() would launch
  auto kernel = detail::bulk_then_execute_kernel<block_dimension,Function,T,ResultFactory,OuterFactory,InnerFactory>::value;

  // get the kernel's attributes
  cudaFuncAttributes attr;
  detail::throw_on_error(cudaFuncGetAttributes(&attr, kernel), "cuda::detail::max_block_size_of_bulk_then_execute_concurrent_kernel(): CUDA error after cudaFuncGetAttributes()");

  // return the attribute of interest
  return attr.maxThreadsPerBlock;
}

template<class Function, class Shape, class T, class ResultFactory, class OuterFactory, class InnerFactory>
__host__ __device__
int max_grid_size_of_bulk_then_execute_concurrent_kernel(const agency::cuda::device_id& device, const Function& f, Shape block_dim, const agency::cuda::async_future<T>& predecessor, const ResultFactory& result_factory, const OuterFactory& outer_factory, const InnerFactory& inner_factory)
{
  const size_t block_dimension = agency::detail::shape_size<Shape>::value;
  constexpr auto kernel = detail::bulk_then_execute_kernel<block_dimension,Function,T,ResultFactory,OuterFactory,InnerFactory>::value;

  int max_active_blocks_per_multiprocessor = 0;
  detail::throw_on_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_multiprocessor, kernel, agency::detail::shape_cast<int>(block_dim), device.native_handle()), "cuda::detail::max_grid_size_of_bulk_then_execute_concurrent_kernel(): CUDA error after cudaOccupancyMaxActiveBlocksPerMultiprocessor()");

  int num_multiprocessors = 0;
  detail::throw_on_error(cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, device.native_handle()), "cuda::detail::max_grid_size_of_bulk_then_execute_concurrent_kernel(): CUDA error after cudaDeviceGetAttribute()");

  return max_active_blocks_per_multiprocessor * num_multiprocessors;
}


} // end detail
} // end cuda
} // end agency

