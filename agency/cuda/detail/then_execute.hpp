#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/detail/stream.hpp>
#include <agency/cuda/detail/event.hpp>
#include <agency/cuda/detail/asynchronous_state.hpp>
#include <agency/cuda/future.hpp>
#include <agency/detail/factory.hpp>
#include <agency/functional.hpp>
#include <agency/detail/shape_cast.hpp>
#include <memory>


// XXX next step: eliminate this file entirely

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Container, class Function, class Shape, class IndexFunction, class T>
__host__ __device__
void* then_execute_kernel(const Function& f, const Shape& shape, const IndexFunction& index_function, const future<T>& fut, const gpu_id& gpu)
{
  auto outer_factory = agency::detail::unit_factory{};
  auto inner_factory = agency::detail::unit_factory{};
  auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};

  return fut.template bulk_then_kernel<Container>(g, shape, index_function, outer_factory, inner_factory, gpu);
}


} // end detail
} // end cuda
} // end agency

