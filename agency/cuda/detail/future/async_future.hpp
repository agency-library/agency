/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/future/event.hpp>
#include <agency/cuda/detail/future/asynchronous_state.hpp>
#include <agency/cuda/detail/future/continuation.hpp>
#include <agency/cuda/detail/on_chip_shared_parameter.hpp>
#include <agency/cuda/gpu.hpp>
#include <agency/detail/unit.hpp>
#include <agency/functional.hpp>
#include <agency/future.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/shape_cast.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class T, class... Args>
struct is_constructible_or_void
  : std::integral_constant<
      bool,
      std::is_constructible<T,Args...>::value ||
      (std::is_void<T>::value && (sizeof...(Args) == 0))
    >
{};


// declare this so future may befriend it
template<class Shape, class Index, class ThisIndexFunction>
class basic_grid_executor;

template<class U>
using element_type_is_not_unit = std::integral_constant<
  bool,
  !std::is_same<typename std::pointer_traits<U>::element_type, agency::detail::unit>::value
>;


// XXX should use empty base class optimization for this class because any of these members could be empty types
//     a simple way to apply this operation would be to derive this class from a tuple of its members, since tuple already applies EBO
// XXX should try to find a way to take an InnerParameterPointer instead of InnerFactory to make the way all the parameters are handled uniformly
// XXX the problem is that the inner parameter needs to know who the leader is, and that info isn't easily passed through pointer dereference syntax
// XXX it would be nice to refactor this functor such that IndexFunction was not a template parameter
//     any reindexing of the CUDA built-ins would happen inside of Function
template<class ContainerPointer, class Function, class IndexFunction, class PastParameterPointer, class OuterParameterPointer, class InnerFactory>
struct bulk_then_functor
{
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
bulk_then_functor<ContainerPointer,Function,IndexFunction,PastParameterPointer,OuterParameterPointer,InnerFactory>
  make_bulk_then_functor(ContainerPointer container_ptr, Function f, IndexFunction index_function, PastParameterPointer past_param_ptr, OuterParameterPointer outer_param_ptr, InnerFactory inner_factory)
{
  return bulk_then_functor<ContainerPointer,Function,IndexFunction,PastParameterPointer,OuterParameterPointer,InnerFactory>{container_ptr, f, index_function, past_param_ptr, outer_param_ptr, inner_factory};
}


} // end detail


// forward declaration for async_future<T>'s benefit
template<class T>
class shared_future;


template<class T>
class async_future
{
  private:
    detail::event event_;
    detail::asynchronous_state<T> state_;

  public:
    __host__ __device__
    async_future() = default;

    __host__ __device__
    async_future(async_future&& other)
      : async_future()
    {
      event_.swap(other.event_);
      state_.swap(other.state_);
    } // end async_future()

    __host__ __device__
    async_future &operator=(async_future&& other)
    {
      event_.swap(other.event_);
      state_.swap(other.state_);
      return *this;
    } // end operator=()

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<
                 detail::asynchronous_state<T>,
                 detail::asynchronous_state<U>&&
               >::value
             >::type>
    __host__ __device__
    async_future(async_future<U>&& other)
      : event_(),
        state_(std::move(other.state_))
    {
      event_.swap(other.event_);
    } // end async_future()

    __host__ __device__
    void wait() const
    {
      event_.wait();
    } // end wait()

    __host__ __device__
    T get()
    {
      wait();

      return state_.get();
    } // end get()

    __host__ __device__
    bool valid() const
    {
      return event_.valid() && state_.valid();
    } // end valid()

    __host__ __device__
    bool is_ready() const
    {
      return event_.is_ready();
    } // end is_ready()

    __host__ __device__
    detail::event& event()
    {
      return event_;
    } // end event()

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    static async_future make_ready(Args&&... args)
    {
      detail::event ready_event(detail::event::construct_ready);

      return async_future(std::move(ready_event), std::forward<Args>(args)...);
    }

    __host__ __device__
    auto data() const -> decltype(state_.data())
    {
      return state_.data();
    }

    template<class Function>
    __host__ __device__
    async_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        async_future
      >
    >
      then(Function f)
    {
      // create state for the continuation's result
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type,async_future>;
      detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready);

      // tuple up f's input state
      auto unfiltered_pointer_tuple = agency::detail::make_tuple(data());

      // filter void states
      auto pointer_tuple = agency::detail::tuple_filter<detail::element_type_is_not_unit>(unfiltered_pointer_tuple);

      // make a function implementing the continuation
      auto continuation = detail::make_continuation(std::forward<Function>(f), result_state.data(), pointer_tuple);

      // launch the continuation
      detail::event next_event = event().then_and_invalidate(std::move(continuation), dim3{1}, dim3{1}, 0);

      // return the continuation's future
      return async_future<result_type>(std::move(next_event), std::move(result_state));
    }

    // XXX the implementation of async_future<T>::share() is in shared_future.hpp
    //     because it needs access to shared_future<T>'s class definition
    shared_future<T> share();

  // XXX stuff beneath here should be private but the implementation of when_all_execute_and_select() uses it
  //private:
    template<class U>
    __host__ __device__
    static U get_ref_impl(agency::detail::empty_type_ptr<U> ptr)
    {
      return *ptr;
    }

    template<class U>
    __host__ __device__
    static U& get_ref_impl(U* ptr)
    {
      return *ptr;
    }

    // XXX get_ref() should be a member of asynchronous_state
    __host__ __device__
    auto get_ref() ->
      decltype(get_ref_impl(this->data()))
    {
      wait();
      return get_ref_impl(data());
    }

    // this version of then() leaves the future in a valid state
    // it's used by future
    template<class Function>
    __host__ __device__
    async_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        async_future
      >
    >
      then_and_leave_valid(Function f)
    {
      // create state for the continuation's result
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type,async_future>;
      detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready);

      // tuple up f's input state
      auto unfiltered_pointer_tuple = agency::detail::make_tuple(data());

      // filter void states
      auto pointer_tuple = agency::detail::tuple_filter<detail::element_type_is_not_unit>(unfiltered_pointer_tuple);
      
      // make a function implementing the continuation
      auto continuation = detail::make_continuation(std::forward<Function>(f), result_state.data(), pointer_tuple);

      // launch the continuation
      detail::event next_event = event().then(std::move(continuation), dim3{1}, dim3{1}, 0);

      // return the continuation's future
      return async_future<result_type>(std::move(next_event), std::move(result_state));
    }

    // XXX should think about getting rid of Shape and IndexFunction
    //     and require grid_dim & block_dim
    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __host__ __device__
    async_future<typename std::result_of<Factory(Shape)>::type>
      bulk_then(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, gpu_id gpu)
    {
      using result_type = typename std::result_of<Factory(Shape)>::type;
      detail::asynchronous_state<result_type> result_state(agency::detail::construct_ready, result_factory(shape));
      
      using outer_arg_type = agency::detail::result_of_factory_t<OuterFactory>;
      auto outer_arg = async_future<outer_arg_type>::make_ready(outer_factory());
      
      auto g = detail::make_bulk_then_functor(result_state.data(), f, index_function, data(), outer_arg.data(), inner_factory);

      uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));
      uint3 inner_shape = agency::detail::shape_cast<uint3>(agency::detail::get<1>(shape));

      ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
      ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};
      
      auto next_event = event().then_on_and_invalidate(g, grid_dim, block_dim, 0, gpu.native_handle());
      
      return async_future<result_type>(std::move(next_event), std::move(result_state));
    }

    template<class Function, class Factory, class Shape, class IndexFunction>
    __host__ __device__
    async_future<typename std::result_of<Factory(Shape)>::type>
      bulk_then(Function f, Factory result_factory, Shape shape, IndexFunction index_function, gpu_id gpu)
    {
      auto outer_factory = agency::detail::unit_factory{};
      auto inner_factory = agency::detail::unit_factory{};
      auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};

      return this->bulk_then(g, result_factory, shape, index_function, outer_factory, inner_factory, gpu);
    }

    // these functions returns a pointer to the kernel used to implement the corresponding call to bulk_then()
    // XXX there might not actually be implemented a corresponding bulk_then() for each of these bulk_then_kernel() functions
    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __host__ __device__
    static void* bulk_then_kernel(const Function& f, const Factory& result_factory, const Shape& s, const IndexFunction& index_function, const OuterFactory&, const InnerFactory& inner_factory, const gpu_id&)
    {
      using result_type = typename std::result_of<Factory(Shape)>::type;
      using result_state_type = detail::asynchronous_state<result_type>;
      using outer_future_type = async_future<agency::detail::result_of_factory_t<OuterFactory>>;

      using bulk_then_functor_type = decltype(detail::make_bulk_then_functor(std::declval<result_state_type>().data(), f, index_function, std::declval<async_future>().data(), std::declval<outer_future_type>().data(), inner_factory));

      return detail::event::then_on_kernel<bulk_then_functor_type>();
    }

    template<class Function, class Factory, class Shape, class IndexFunction>
    __host__ __device__
    static void* bulk_then_kernel(const Function& f, const Factory& result_factory, const Shape& s, const IndexFunction& index_function, const gpu_id& gpu)
    {
      auto outer_factory = agency::detail::unit_factory{};
      auto inner_factory = agency::detail::unit_factory{};
      auto g = agency::detail::take_first_two_parameters_and_invoke<Function>{f};

      return bulk_then_kernel(f, result_factory, s, index_function, outer_factory, inner_factory, gpu);
    }

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __host__ __device__
    async_future<typename std::result_of<Factory(Shape)>::type>
      bulk_then_and_leave_valid(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, gpu_id gpu)
    {
      using result_type = typename std::result_of<Factory(Shape)>::type;
      detail::asynchronous_state<result_type> result_state(agency::detail::construct_ready, result_factory(shape));
      
      using outer_arg_type = agency::detail::result_of_factory_t<OuterFactory>;
      auto outer_arg = async_future<outer_arg_type>::make_ready(outer_factory());
      
      auto g = detail::make_bulk_then_functor(result_state.data(), f, index_function, data(), outer_arg.data(), inner_factory);

      uint3 outer_shape = agency::detail::shape_cast<uint3>(agency::detail::get<0>(shape));
      uint3 inner_shape = agency::detail::shape_cast<uint3>(agency::detail::get<1>(shape));

      ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
      ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};
      
      auto next_event = event().then_on(g, grid_dim, block_dim, 0, gpu.native_handle());
      
      return async_future<result_type>(std::move(next_event), std::move(result_state));
    }

    template<class U> friend class async_future;
    template<class Shape, class Index, class ThisIndexFunction> friend class agency::cuda::detail::basic_grid_executor;

    __host__ __device__
    async_future(detail::event&& e, detail::asynchronous_state<T>&& state)
      : event_(std::move(e)), state_(std::move(state))
    {}

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    async_future(detail::event&& e, Args&&... ready_args)
      : async_future(std::move(e), detail::asynchronous_state<T>(agency::detail::construct_ready, std::forward<Args>(ready_args)...))
    {}

    // implement swap to avoid depending on thrust::swap
    template<class U>
    __host__ __device__
    static void swap(U& a, U& b)
    {
      U tmp{a};
      a = b;
      b = tmp;
    }

    template<class... Types>
    friend __host__ __device__
    async_future<
      agency::detail::when_all_result_t<
        async_future<Types>...
      >
    >
    when_all(async_future<Types>&... futures);
};


inline __host__ __device__
async_future<void> make_ready_async_future()
{
  return async_future<void>::make_ready();
} // end make_ready_async_future()

template<class T>
__host__ __device__
async_future<T> make_ready_async_future(T&& val)
{
  return async_future<T>::make_ready(std::forward<T>(val));
}


namespace detail
{


template<class Result>
struct when_all_functor
{
  template<class... Args>
  __host__ __device__
  Result operator()(Args&... args)
  {
    return Result(std::move(args)...);
  }
};


} // end detail


template<class... Types>
__host__ __device__
async_future<
  agency::detail::when_all_result_t<
    async_future<Types>...
  >
>
when_all(async_future<Types>&... futures)
{
  // join the events
  detail::event when_all_ready = detail::when_all_events_are_ready(futures.event()...);

  using result_type = agency::detail::when_all_result_t<
    async_future<Types>...
  >;

  detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready);

  // tuple up the input states
  auto unfiltered_pointer_tuple = agency::detail::make_tuple(futures.data()...);

  // filter void states
  auto pointer_tuple = agency::detail::tuple_filter<detail::element_type_is_not_unit>(unfiltered_pointer_tuple);

  // make a function implementing the continuation
  auto continuation = detail::make_continuation(detail::when_all_functor<result_type>{}, result_state.data(), pointer_tuple);

  // launch the continuation
  detail::event next_event = when_all_ready.then_and_invalidate(continuation, dim3{1}, dim3{1}, 0);

  // return the continuation's future
  return async_future<result_type>(std::move(next_event), std::move(result_state));
}


template<class T>
__host__ __device__
async_future<T> when_all(async_future<T>& future)
{
  return std::move(future);
}


} // end namespace cuda
} // end namespace agency

