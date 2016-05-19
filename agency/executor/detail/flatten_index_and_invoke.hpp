#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/factory.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/shape_tuple.hpp>
#include <agency/detail/shape.hpp>
#include <agency/detail/index.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class ShapeTuple>
using flattened_shape_type_t = merge_front_shape_elements_t<ShapeTuple>;


// XXX might not want to use a alias template here
template<class IndexTuple>
using flattened_index_type_t = merge_front_shape_elements_t<IndexTuple>;




// flatten_index_and_invoke is a helper functor for flattened_executor The idea
// is it takes an index received from flattened_executor's base_executor and
// projects the index into flattened_executor's index space.  This projection
// operation can produce indices outside of flattened_executor's index space.
// When the projected index is outside of the index space, we say that the
// function is not defined.
//
// When the result of the invoked function is void, the result of
// project_index_and_invoke is void. Otherwise, the result is optionally a
// struct containing the result of the function and the projected index where
// it was invoked. If the function is not defined at the projected index, the result
// of project_index_and_invoke is an empty optional.


template<class Index, class Function, class Shape>
struct flatten_index_and_invoke_base
{
  using index_type = Index;
  using shape_type = Shape;

  using flattened_index_type = flattened_index_type_t<Index>;
  using flattened_shape_type = flattened_shape_type_t<Shape>;

  mutable Function     f_;
  shape_type           shape_;
  flattened_shape_type flattened_shape_;

  __AGENCY_ANNOTATION
  flatten_index_and_invoke_base(const Function& f, shape_type shape, flattened_shape_type flattened_shape)
    : f_(f),
      shape_(shape),
      flattened_shape_(flattened_shape)
  {}

  // this type stores the result of f_(index)
  template<class T>
  struct value_and_index
  {
    T value;
    flattened_index_type index;
  };

  template<class T>
  __AGENCY_ANNOTATION
  value_and_index<typename std::decay<T>::type> make_value_and_index(T&& value, flattened_index_type idx) const
  {
    return value_and_index<typename std::decay<T>::type>{std::forward<T>(value), idx};
  }

  // this is the type of result returned by f_
  template<class... Args>
  using result_of_function_t = result_of_t<Function(flattened_index_type,Args...)>;

  template<class T>
  using void_or_optionally_value_and_index_t = typename std::conditional<
    std::is_void<T>::value,
    void,
    experimental::optional<value_and_index<T>>
  >::type;

  // this is the type of result returned by this functor
  template<class... Args>
  using result_t = void_or_optionally_value_and_index_t<result_of_function_t<Args...>>;

  __AGENCY_ANNOTATION
  flattened_index_type flatten_index(const Index& idx) const
  {
    return detail::merge_front_index_elements(idx, shape_);
  }

  __AGENCY_ANNOTATION
  bool in_domain(const flattened_index_type& idx) const
  {
    // idx is in the domain of f_ if idx is contained within the
    // axis-aligned bounded box from extremal corners at the origin
    // and flattened_shape_. the "hyper-interval" is half-open, so
    // the origin is contained within the box but the corner at
    // flattened_shape_ is not.
    return detail::is_bounded_by(idx, flattened_shape_);
  }

  // when f_(idx) has no result, we just return void
  template<class... Args,
           class = typename std::enable_if<
             std::is_void<
               result_of_function_t<Args&&...>
             >::value
           >::type
          >
  __AGENCY_ANNOTATION
  void impl(const Index& idx, Args&&... args) const
  {
    flattened_index_type flattened_idx = flatten_index(idx);

    if(in_domain(flattened_idx))
    {
      f_(flattened_idx, std::forward<Args>(args)...);
    }
  }

  // when f_(idx) has a result, we optionally return f_'s result and the index,
  // but only at points where f_(idx) is defined
  template<class... Args,
           class = typename std::enable_if<
             !std::is_void<
               result_of_function_t<Args&&...>
             >::value
           >::type
          >
  __AGENCY_ANNOTATION
  result_t<Args&&...>
    impl(const Index& idx, Args&&... args) const
  {
    flattened_index_type flattened_idx = flatten_index(idx);

    if(in_domain(flattened_idx))
    {
      return make_value_and_index(f_(flattened_idx, std::forward<Args>(args)...), flattened_idx);
    }

    return experimental::nullopt;
  }
};


// this handles the case when the dependency future's type is not void
template<class Index, class PastParameterT, class Function, class Shape>
struct flatten_index_and_invoke : flatten_index_and_invoke_base<Index,Function,Shape>
{
  using flatten_index_and_invoke_base<Index,Function,Shape>::flatten_index_and_invoke_base;

  template<class T, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(const Index& idx, PastParameterT& past_parameter, T& outer_shared_parameter, unit, Args&... inner_shared_parameters) const ->
    decltype(this->impl(idx, past_parameter, outer_shared_parameter, inner_shared_parameters...))
  {
    // ignore the unit parameter and forward the rest
    return this->impl(idx, past_parameter, outer_shared_parameter, inner_shared_parameters...);
  }
};


// this handles the case when the dependency future's type is void
template<class Index, class Function, class Shape>
struct flatten_index_and_invoke<Index,void,Function,Shape> : flatten_index_and_invoke_base<Index,Function,Shape>
{
  using flatten_index_and_invoke_base<Index,Function,Shape>::flatten_index_and_invoke_base;

  template<class T, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(const Index& idx, T& outer_shared_parameter, unit, Args&... inner_shared_parameters) const ->
    decltype(this->impl(idx, outer_shared_parameter, inner_shared_parameters...))
  {
    // ignore the unit parameter and forward the rest
    return this->impl(idx, outer_shared_parameter, inner_shared_parameters...);
  }
};


template<class Index, class PastParameterT, class Function, class Shape>
__AGENCY_ANNOTATION
flatten_index_and_invoke<Index,PastParameterT,Function,Shape>
  make_flatten_index_and_invoke(Function f,
                                Shape higher_dimensional_shape,
                                typename flatten_index_and_invoke<Index,PastParameterT,Function,Shape>::flattened_shape_type lower_dimensional_shape)
{
  return flatten_index_and_invoke<Index,PastParameterT,Function,Shape>{f,higher_dimensional_shape,lower_dimensional_shape};
}


} // end detail
} // end agency

