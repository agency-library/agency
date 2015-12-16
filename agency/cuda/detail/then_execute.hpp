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
  __device__ static inline void impl(Function f, const Index &idx, T1& container, agency::detail::unit, T3& outer_param, T4& inner_param)
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


template<class Container, class Function, class Shape, class IndexFunction, class T, class OuterFactory, class InnerFactory>
__host__ __device__
future<Container> then_execute(Function f, Shape shape, IndexFunction index_function, future<T>& fut, OuterFactory outer_factory, InnerFactory inner_factory)
{
  detail::stream stream = std::move(fut.stream());
  
  detail::asynchronous_state<Container> result_state(agency::detail::construct_ready, shape);
  
  using outer_arg_type = agency::detail::result_of_factory_t<OuterFactory>;
  auto outer_arg = cuda::make_ready_future<outer_arg_type>(outer_factory());
  
  auto g = detail::make_then_execute_functor(result_state.data(), f, index_function, fut.data(), outer_arg.data(), inner_factory);

  uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));
  uint3 inner_shape = agency::detail::shape_cast<uint3>(agency::detail::get<1>(shape));

  ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
  ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};
  
  auto next_event = fut.event().then_on(g, grid_dim, block_dim, 0, stream.native_handle());
  
  return future<Container>(std::move(stream), std::move(next_event), std::move(result_state));
}


// this function returns a pointer to the kernel used to implement then_execute()
template<class Container, class Function, class Shape, class IndexFunction, class T, class OuterFactory, class InnerFactory>
__host__ __device__
void* then_execute_kernel(const Function& f, const Shape& s, const IndexFunction& index_function, const future<T>& fut, const OuterFactory& outer_factory, const InnerFactory& inner_factory)
{
  using result_state_type = detail::asynchronous_state<Container>;
  using outer_future_type = future<agency::detail::result_of_factory_t<OuterFactory>>;

  using then_execute_functor_type = decltype(detail::make_then_execute_functor(std::declval<result_state_type>().data(), f, index_function, fut.data(), std::declval<outer_future_type>().data(), inner_factory));

  return detail::event::then_kernel<then_execute_functor_type>();
}



template<class Container, class Function, class Shape, class IndexFunction, class T>
__host__ __device__
future<Container> then_execute(Function f, Shape shape, IndexFunction index_function, future<T>& fut)
{
  auto outer_factory = agency::detail::unit_factory{};
  auto inner_factory = agency::detail::unit_factory{};
  auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};

  return detail::then_execute<Container>(g, shape, index_function, fut, outer_factory, inner_factory);
}


template<class Container, class Function, class Shape, class IndexFunction, class T>
__host__ __device__
void* then_execute_kernel(const Function& f, const Shape& shape, const IndexFunction& index_function, const future<T>& fut)
{
  auto outer_factory = agency::detail::unit_factory{};
  auto inner_factory = agency::detail::unit_factory{};
  auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};

  return detail::then_execute_kernel<Container>(g, shape, index_function, fut, outer_factory, inner_factory);
}


} // end detail
} // end cuda
} // end agency

