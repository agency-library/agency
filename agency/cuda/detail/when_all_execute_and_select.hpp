#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/future.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/device.hpp>
#include <type_traits>
#include <memory>
#include <utility>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function, class IndexFunction, class OuterArgumentPointer, class InnerFactory, class... DependencyPointers>
struct when_all_execute_functor
{
  Function                                     f_;
  IndexFunction                                index_function_;
  agency::detail::tuple<DependencyPointers...> dependency_ptrs_;
  OuterArgumentPointer                         outer_arg_ptr_;
  InnerFactory                                 inner_factory_;

  __host__ __device__
  when_all_execute_functor(Function f, IndexFunction index_function, OuterArgumentPointer outer_arg_ptr, InnerFactory inner_factory, DependencyPointers... dependency_ptrs)
    : f_(f),
      index_function_(index_function),
      dependency_ptrs_(dependency_ptrs...),
      outer_arg_ptr_(outer_arg_ptr),
      inner_factory_(inner_factory)
  {}

  template<size_t... Indices>
  __device__
  void impl(agency::detail::index_sequence<Indices...>)
  {
    auto idx = index_function_();

    // XXX i don't think we're doing the leader calculation in a portable way
    //     we need a way to compare idx to the origin idx to figure out if this invocation represents the CTA leader
    on_chip_shared_parameter<InnerFactory> inner_param(idx[1] == 0, inner_factory_);

    // convert the references to raw references before passing them to f_
    f_(idx,
       static_cast<typename std::pointer_traits<DependencyPointers>::element_type&>(*agency::detail::get<Indices>(dependency_ptrs_))...,
       static_cast<typename std::pointer_traits<OuterArgumentPointer>::element_type&>(*outer_arg_ptr_),
       inner_param.get());
  }

  __device__
  void operator()()
  {
    impl(agency::detail::index_sequence_for<DependencyPointers...>());
  }
};


template<class Function, class IndexFunction, class OuterArgumentPointer, class InnerFactory, class... DependencyPointers>
__host__ __device__
when_all_execute_functor<Function, IndexFunction, OuterArgumentPointer, InnerFactory, DependencyPointers...>
  make_when_all_execute_functor(Function f, IndexFunction index_function, OuterArgumentPointer outer_arg_ptr, InnerFactory inner_factory, DependencyPointers... dependency_ptrs)
{
  return when_all_execute_functor<Function, IndexFunction, OuterArgumentPointer, InnerFactory, DependencyPointers...>(f, index_function, outer_arg_ptr, inner_factory, dependency_ptrs...);
}


// metafunction returns whether or not the given Future's nested value_type member type is not void
// the template parameter FutureOrFutureReference may either be a Future type or a reference to a Future type
template<class FutureOrFutureReference>
struct value_type_is_not_void
  : std::integral_constant<
      bool,
      !std::is_void<
        typename agency::future_traits<
          typename std::decay<FutureOrFutureReference>::type
        >::value_type
      >::value
    >
{};


// this functor takes a number of pointers
// it assigns a result_type to the first pointer's pointee
// the result is constructed by moving the other pointees into result_type's constructor
template<class ResultPointer, class... Pointers>
struct move_construct_result_functor
{
  ResultPointer result_ptr_;
  agency::cuda::detail::tuple<Pointers...> ptrs_;

  using result_type = typename std::pointer_traits<ResultPointer>::element_type;

  __AGENCY_ANNOTATION
  move_construct_result_functor(ResultPointer result_ptr, Pointers... ptrs)
    : result_ptr_(result_ptr),
      ptrs_(ptrs...)
  {}

  template<size_t... Indices>
  __AGENCY_ANNOTATION
  inline void impl(agency::detail::index_sequence<Indices...>)
  {
    *result_ptr_ = result_type(std::move(*agency::detail::get<Indices>(ptrs_))...);
  }

  __AGENCY_ANNOTATION
  inline void operator()()
  {
    impl(agency::detail::index_sequence_for<Pointers...>());
  }
};


template<class ResultPointer, class... Pointers>
__AGENCY_ANNOTATION
move_construct_result_functor<ResultPointer,Pointers...>
  make_move_construct_result_functor(ResultPointer result_ptr, Pointers... ptrs)
{
  return move_construct_result_functor<ResultPointer,Pointers...>(result_ptr,ptrs...);
}


// this function launches a kernel dependent on the given event
// to move construct the object pointed to by ptr
// the pointees of ptrs are moved into the result's constructor
template<class Pointer, class... Pointers>
__host__ __device__
agency::cuda::async_future<
  agency::detail::when_all_result_t<
    agency::cuda::async_future<
      typename std::pointer_traits<Pointer>::element_type
    >,
    agency::cuda::async_future<
      typename std::pointer_traits<Pointers>::element_type
    >...
  >
>
  move_construct_result(agency::cuda::detail::event& dependency, Pointer ptr, Pointers... ptrs)
{
  using result_type = agency::detail::when_all_result_t<
    agency::cuda::async_future<typename std::pointer_traits<Pointer>::element_type>,
    agency::cuda::async_future<typename std::pointer_traits<Pointers>::element_type>...
  >;

  // create a state to hold the result
  agency::cuda::detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready);

  // create a function to do the move construction
  auto f = make_move_construct_result_functor(result_state.data(), ptr, ptrs...);

  // launch the function
  agency::cuda::detail::event result_event = dependency.then_and_invalidate(f, dim3{1}, dim3{1}, 0);

  return agency::cuda::async_future<result_type>(std::move(result_event), std::move(result_state));
}


inline __host__ __device__
agency::cuda::async_future<void>
  move_construct_result(agency::cuda::detail::event& dependency)
{
  return agency::cuda::async_future<void>(std::move(dependency), agency::cuda::detail::asynchronous_state<void>(agency::detail::construct_not_ready));
}


template<size_t... Indices, class TupleOfFutures>
__host__ __device__
auto move_construct_result_from_tuple_of_futures(agency::detail::index_sequence<Indices...>, agency::cuda::detail::event& dependency, TupleOfFutures& futures)
  -> decltype(move_construct_result(dependency, agency::detail::get<Indices>(futures).data()...))
{
  return move_construct_result(dependency, agency::detail::get<Indices>(futures).data()...);
}


template<class Function, class Shape, class IndexFunction, class OuterArgPointer, class InnerFactory, class... Pointers>
__host__ __device__
event launch_when_all_execute_operation_impl(event& dependency,
                                             Function f,
                                             Shape shape,
                                             IndexFunction index_function,
                                             OuterArgPointer outer_arg_ptr,
                                             InnerFactory inner_factory,
                                             Pointers... ptrs)
{
  // make a function implementing the operation
  auto continuation = make_when_all_execute_functor(f, index_function, outer_arg_ptr, inner_factory, ptrs...);

  // convert the shape to CUDA types
  agency::uint3 outer_shape = agency::detail::shape_cast<agency::uint3>(agency::detail::get<0>(shape));
  agency::uint3 inner_shape = agency::detail::shape_cast<agency::uint3>(agency::detail::get<1>(shape));

  ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
  ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};

  // launch the continuation
  return dependency.then_and_invalidate(continuation, grid_dim, block_dim, 0);
}


template<size_t... Indices, class Function, class Shape, class IndexFunction, class OuterArgumentPointer, class InnerFactory, class TupleOfNonVoidFutures>
__host__ __device__
event launch_when_all_execute_operation(agency::detail::index_sequence<Indices...>,
                                        event& dependency,
                                        Function f,
                                        Shape shape,
                                        IndexFunction index_function,
                                        OuterArgumentPointer outer_arg_ptr,
                                        InnerFactory inner_factory,
                                        TupleOfNonVoidFutures& futures)
{
  // unpack the futures and pass their data pointers to launch_when_all_execute_operation_impl()
  return cuda::detail::launch_when_all_execute_operation_impl(dependency, f, shape, index_function, outer_arg_ptr, inner_factory, agency::detail::get<Indices>(futures).data()...);
}


template<class IndexSequence, class TypeList>
struct when_all_execute_and_select_result_from_type_list;

template<size_t... Indices, class TypeList>
struct when_all_execute_and_select_result_from_type_list<agency::detail::index_sequence<Indices...>, TypeList>
{
  using unfiltered_type_list = agency::detail::type_list<
    agency::detail::type_list_element<Indices,TypeList>...
  >;

  template<class T>
  struct is_not_void : std::integral_constant<bool, !std::is_void<T>::value> {};

  using non_void_types = agency::detail::type_list_filter<is_not_void,unfiltered_type_list>;

  using type = agency::detail::tuple_or_single_type_or_void_from_type_list_t<non_void_types>;
};


template<class IndexSequence, class TupleOfFutures>
struct when_all_execute_and_select_result
{
  // get the tuple's type_list of futures
  using future_types = agency::detail::tuple_elements<TupleOfFutures>;

  template<class Future>
  struct future_value_type
  {
    using type = typename agency::future_traits<Future>::value_type;
  };

  // get each future's value_type
  using value_types = agency::detail::type_list_map<future_value_type, future_types>;

  using type = typename when_all_execute_and_select_result_from_type_list<IndexSequence, value_types>::type;
};

template<class IndexSequence, class TupleOfFutures>
using when_all_execute_and_select_result_t = typename when_all_execute_and_select_result<IndexSequence, TupleOfFutures>::type;


template<size_t... SelectedIndices, size_t... TupleIndices, class Function, class Shape, class IndexFunction, class TupleOfFutures, class OuterFactory, class InnerFactory>
__host__ __device__
async_future<when_all_execute_and_select_result_t<agency::detail::index_sequence<SelectedIndices...>, TupleOfFutures>>
  when_all_execute_and_select_impl(agency::detail::index_sequence<SelectedIndices...>,
                                   agency::detail::index_sequence<TupleIndices...>,
                                   Function f,
                                   Shape shape,
                                   IndexFunction index_function,
                                   TupleOfFutures tuple_of_futures,
                                   OuterFactory outer_factory,
                                   InnerFactory inner_factory,
                                   const device_id& device)
{
  // create a future to contain the outer argument
  using outer_arg_type = agency::detail::result_of_t<OuterFactory()>;
  auto outer_arg_future = agency::cuda::make_ready_async_future<outer_arg_type>(outer_factory());

  // join the events
  event when_all_ready = cuda::detail::when_all_events_are_ready(device, outer_arg_future.event(), agency::detail::get<TupleIndices>(tuple_of_futures).event()...);

  // get a view of the non-void futures
  auto view_of_non_void_futures = agency::detail::tuple_filter_view<value_type_is_not_void>(tuple_of_futures);

  // launch the main operation
  auto when_all_execute_event = cuda::detail::launch_when_all_execute_operation(agency::detail::make_tuple_indices(view_of_non_void_futures), when_all_ready, f, shape, index_function, outer_arg_future.data(), inner_factory, view_of_non_void_futures);

  // get a view of the selected futures
  auto view_of_selected_futures = agency::detail::forward_as_tuple(agency::detail::get<SelectedIndices>(tuple_of_futures)...);

  // get a view of the selected futures which are non-void
  auto view_of_selected_non_void_futures = agency::detail::tuple_filter_view<value_type_is_not_void>(view_of_selected_futures);

  // XXX we need to figure out how to safely end the futures' lifetimes
  //     we can't destroy them before their values have been moved into the result future
  //     we need to garbage collect them somehow
  //     or we need to take their data pointer and make the result construction kernel the owner

  // XXX to do this the right way, we'd move each future's pointer into the result construction kernel
  //     the agent that executes that operation would take ownership of the unique_ptrs and they would naturally get destroyed when that agent ended its computation
  //     we can't do that because that memory cannot be deallocated by a __device__ function
  // XXX we should implement a better memory allocator that can be used uniformly in __host__ & __device__ code

  return cuda::detail::move_construct_result_from_tuple_of_futures(agency::detail::make_tuple_indices(view_of_selected_non_void_futures), when_all_execute_event, view_of_selected_non_void_futures);
}


template<size_t... SelectedIndices,
         class Function,
         class Shape,
         class IndexFunction,
         class TupleOfFutures,
         class OuterFactory,
         class InnerFactory>
__host__ __device__
async_future<when_all_execute_and_select_result_t<agency::detail::index_sequence<SelectedIndices...>, agency::detail::decay_t<TupleOfFutures>>>
  when_all_execute_and_select(Function f,
                              Shape shape,
                              IndexFunction index_function,
                              TupleOfFutures&& tuple_of_futures,
                              OuterFactory outer_factory,
                              InnerFactory inner_factory,
                              const device_id& device)
{
  // XXX we should static_assert that SelectedIndices are unique and in the correct range

  return cuda::detail::when_all_execute_and_select_impl(agency::detail::index_sequence<SelectedIndices...>(),
                                                        agency::detail::make_tuple_indices(tuple_of_futures),
                                                        f,
                                                        shape,
                                                        index_function,
                                                        std::move(tuple_of_futures),
                                                        outer_factory,
                                                        inner_factory,
                                                        device);
}


} // end detail
} // end cuda
} // end agency

