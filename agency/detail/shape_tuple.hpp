#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/arithmetic_tuple_facade.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/make_tuple_if_not_nested.hpp>

namespace agency
{
namespace detail
{


// shape_tuple can't just be an alias for a particular kind of tuple
// because it also requires arithmetic operators
template<class... Shapes>
class shape_tuple :
  public agency::detail::tuple<Shapes...>,
  public arithmetic_tuple_facade<shape_tuple<Shapes...>>
{
  public:
    using agency::detail::tuple<Shapes...>::tuple;
};

template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Shape1,
         class Shape2>
struct nested_shape
{
  using type = decltype(
    detail::tuple_cat(
      detail::make_tuple_if_not_nested<ExecutionCategory1>(std::declval<Shape1>()),
      detail::make_tuple_if_not_nested<ExecutionCategory2>(std::declval<Shape2>())
    )
  );
};


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Shape1,
         class Shape2>
using nested_shape_t = typename nested_shape<
  ExecutionCategory1,
  ExecutionCategory2,
  Shape1,
  Shape2
>::type;


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Shape1,
         class Shape2>
nested_shape_t<ExecutionCategory1,ExecutionCategory2,Shape1,Shape2> make_nested_shape(const Shape1& outer_shape, const Shape2& inner_shape)
{
  return detail::tuple_cat(
    detail::make_tuple_if_not_nested<ExecutionCategory1>(outer_shape),
    detail::make_tuple_if_not_nested<ExecutionCategory2>(inner_shape)
  );
}

} // end detail
} // end agency


namespace __tu
{

// tuple_traits specializations

template<class... Shapes>
struct tuple_traits<agency::detail::shape_tuple<Shapes...>>
  : __tu::tuple_traits<agency::detail::tuple<Shapes...>>
{
  using tuple_type = agency::detail::tuple<Shapes...>;
}; // end tuple_traits


} // end __tu


namespace std
{


template<class... Shapes>
struct tuple_size<agency::detail::shape_tuple<Shapes...>> : std::tuple_size<agency::detail::tuple<Shapes...>> {};

template<size_t i, class... Shapes>
struct tuple_element<i,agency::detail::shape_tuple<Shapes...>> : std::tuple_element<i,agency::detail::tuple<Shapes...>> {};


} // end namespace std

