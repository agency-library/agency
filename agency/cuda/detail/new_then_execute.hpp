#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/detail/event.hpp>
#include <agency/cuda/detail/asynchronous_state.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/gpu.hpp>
#include <agency/detail/factory.hpp>
#include <agency/functional.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/cuda/detail/then_execute.hpp>
#include <memory>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function, class Factory, class Shape, class IndexFunction, class T, class OuterFactory, class InnerFactory>
__host__ __device__
future<typename std::result_of<Factory(Shape)>::type>
  new_then_execute(Function f, Factory result_factory, Shape shape, IndexFunction index_function, future<T>& fut, OuterFactory outer_factory, InnerFactory inner_factory, gpu_id gpu)
{
  using result_type = typename std::result_of<Factory(Shape)>::type;
  detail::asynchronous_state<result_type> result_state(agency::detail::construct_ready, result_factory(shape));
  
  using outer_arg_type = agency::detail::result_of_factory_t<OuterFactory>;
  auto outer_arg = cuda::make_ready_future<outer_arg_type>(outer_factory());
  
  auto g = detail::make_then_execute_functor(result_state.data(), f, index_function, fut.data(), outer_arg.data(), inner_factory);

  uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));
  uint3 inner_shape = agency::detail::shape_cast<uint3>(agency::detail::get<1>(shape));

  ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
  ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};
  
  auto next_event = fut.event().then_on(g, grid_dim, block_dim, 0, gpu.native_handle());
  
  return future<result_type>(std::move(next_event), std::move(result_state));
}


// this function returns a pointer to the kernel used to implement then_execute()
template<class Function, class Factory, class Shape, class IndexFunction, class T, class OuterFactory, class InnerFactory>
__host__ __device__
void* new_then_execute_kernel(const Function& f, const Factory& result_factory, const Shape& s, const IndexFunction& index_function, const future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory, const gpu_id&)
{
  using result_type = typename std::result_of<Factory(Shape)>::type;
  using result_state_type = detail::asynchronous_state<result_type>;
  using outer_future_type = future<agency::detail::result_of_factory_t<OuterFactory>>;

  using then_execute_functor_type = decltype(detail::make_then_execute_functor(std::declval<result_state_type>().data(), f, index_function, fut.data(), std::declval<outer_future_type>().data(), inner_factory));

  return detail::event::then_on_kernel<then_execute_functor_type>();
}



template<class Function, class Factory, class Shape, class IndexFunction, class T>
__host__ __device__
future<typename std::result_of<Factory(Shape)>::type>
  new_then_execute(Function f, Factory result_factory, Shape shape, IndexFunction index_function, future<T>& fut, gpu_id gpu)
{
  auto outer_factory = agency::detail::unit_factory{};
  auto inner_factory = agency::detail::unit_factory{};
  auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};

  return detail::new_then_execute(g, result_factory, shape, index_function, fut, outer_factory, inner_factory, gpu);
}


template<class Function, class Factory, class Shape, class IndexFunction, class T>
__host__ __device__
void* new_then_execute_kernel(const Function& f, const Factory& result_factory, const Shape& shape, const IndexFunction& index_function, const future<T>& fut, const gpu_id& gpu)
{
  auto outer_factory = agency::detail::unit_factory{};
  auto inner_factory = agency::detail::unit_factory{};
  auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};

  return detail::new_then_execute_kernel(g, result_factory, shape, index_function, fut, outer_factory, inner_factory, gpu);
}


} // end detail
} // end cuda
} // end agency

