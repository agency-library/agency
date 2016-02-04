#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/optional.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/shape_tuple.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class TypeList>
struct merge_first_two_types;

template<class T1, class T2, class... Types>
struct merge_first_two_types<type_list<T1,T2,Types...>>
{
  // XXX we probably want to think carefully about what it means two "merge" two arithmetic tuples together
  template<class U1, class U2>
  using merge_types_t = typename std::common_type<U1,U2>::type;

  using type = type_list<merge_types_t<T1,T2>, Types...>;
};


template<class ShapeTuple>
struct flattened_shape_type
{
  // two "flatten" a shape tuple, we merge the last two elements
  using elements = tuple_elements<ShapeTuple>;

  // reverse the type_list, merge the first two types, and then unreverse the result
  using reversed_elements = type_list_reverse<elements>;
  using merged_reversed_elements = typename merge_first_two_types<reversed_elements>::type;
  using merged_elements = type_list_reverse<merged_reversed_elements>;

  // turn the resulting type list into a shape_tuple
  using shape_tuple_type = type_list_instantiate<shape_tuple, merged_elements>;

  // finally, unwrap single-element tuples
  using type = typename std::conditional<
    (std::tuple_size<shape_tuple_type>::value == 1),
    typename std::tuple_element<0,shape_tuple_type>::type,
    shape_tuple_type
  >::type;
};


template<class ShapeTuple>
using flattened_shape_type_t = typename flattened_shape_type<ShapeTuple>::type;


template<class IndexTuple>
struct flattened_index_type
{
  // two "flatten" a index tuple, we merge the last two elements
  using elements = tuple_elements<IndexTuple>;

  // reverse the type_list, merge the first two types, and then unreverse the result
  using reversed_elements = type_list_reverse<elements>;
  using merged_reversed_elements = typename merge_first_two_types<reversed_elements>::type;
  using merged_elements = type_list_reverse<merged_reversed_elements>;

  // turn the resulting type list into a index_tuple
  using index_tuple_type = type_list_instantiate<index_tuple, merged_elements>;

  // finally, unwrap single-element tuples
  using type = typename std::conditional<
    (std::tuple_size<index_tuple_type>::value == 1),
    typename std::tuple_element<0,index_tuple_type>::type,
    index_tuple_type
  >::type;
};

template<class IndexTuple>
using flattened_index_type_t = typename flattened_index_type<IndexTuple>::type;




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
  using result_of_function_t = typename std::result_of<Function(flattened_index_type,Args...)>::type;

  template<class T>
  using void_or_optionally_value_and_index_t = typename std::conditional<
    std::is_void<T>::value,
    void,
    optional<value_and_index<T>>
  >::type;

  // this is the type of result returned by this functor
  template<class... Args>
  using result_t = void_or_optionally_value_and_index_t<result_of_function_t<Args...>>;

  template<size_t... Indices>
  __AGENCY_ANNOTATION
  flattened_index_type flatten_index_impl(detail::index_sequence<Indices...>, const Index& idx) const
  {
    // to flatten a multidimensional index, we take the last two elements of idx and merge them together
    // this merger becomes the last element of the result
    // the remaining elements of idx shift one position right
    constexpr size_t last_idx = std::tuple_size<Index>::value - 1;
    return flattened_index_type(detail::get<Indices>(idx)..., detail::get<last_idx>(shape_) * detail::get<last_idx-1>(idx) + detail::get<last_idx>(idx));
  }

  // XXX WAR nvcc issue with handling empty index_sequence<> given to the function above
  __AGENCY_ANNOTATION
  flattened_index_type flatten_index_impl(detail::index_sequence<>, const Index& idx) const
  {
    return flattened_index_type(detail::get<1>(idx) + detail::get<0>(idx) * detail::get<1>(shape_));
  }

  __AGENCY_ANNOTATION
  flattened_index_type flatten_index(const Index& idx) const
  {
    return flatten_index_impl(detail::make_index_sequence<std::tuple_size<Index>::value - 2>(), idx);
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

    return nullopt;
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

