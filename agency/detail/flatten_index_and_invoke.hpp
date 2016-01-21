#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/optional.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class TypeList>
struct flattened_shape_type;

template<class Shape1, class Shape2, class... Shapes>
struct flattened_shape_type<type_list<Shape1,Shape2,Shapes...>>
{
  // XXX we probably want to think carefully about what it means two "merge" two arithmetic tuples together
  template<class T1, class T2>
  using merge_shapes_t = typename std::common_type<T1,T2>::type;

  using tuple_type = shape_tuple<
    merge_shapes_t<Shape1,Shape2>,
    Shapes...
  >;

  // unwrap single-element tuples
  using type = typename std::conditional<
    (std::tuple_size<tuple_type>::value == 1),
    typename std::tuple_element<0,tuple_type>::type,
    tuple_type
  >::type;
};


template<class ShapeTuple>
using flattened_shape_type_t = typename flattened_shape_type<tuple_elements<ShapeTuple>>::type;


template<class TypeList>
struct flattened_index_type;

template<class Index1, class Index2, class... Indices>
struct flattened_index_type<type_list<Index1,Index2,Indices...>>
{
  // XXX we probably want to think carefully about what it means two "merge" two arithmetic tuples together
  template<class T1, class T2>
  using merge_indices_t = typename std::common_type<T1,T2>::type;

  using tuple_type = index_tuple<
    merge_indices_t<Index1,Index2>,
    Indices...
  >;

  // unwrap single-element tuples
  using type = typename std::conditional<
    (std::tuple_size<tuple_type>::value == 1),
    typename std::tuple_element<0,tuple_type>::type,
    tuple_type
  >::type;
};

template<class IndexTuple>
using flattened_index_type_t = flattened_shape_type_t<IndexTuple>;




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
    // to flatten a multidimensional index, we take the first two elements of idx and merge them together
    // this merger becomes the first element of the result
    // the remaining elements of idx shift one position left
    return flattened_index_type(detail::get<1>(idx) + detail::get<0>(idx) * detail::get<1>(shape_), detail::get<2 + Indices>(idx)...);
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

    if(flattened_idx < flattened_shape_)
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

    if(flattened_idx < flattened_shape_)
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

