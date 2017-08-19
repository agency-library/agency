#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/tuple/detail/arithmetic_tuple_facade.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/make_tuple_if_not_scoped.hpp>

namespace agency
{
namespace detail
{


// shape_tuple can't just be an alias for a particular kind of tuple
// because it also requires arithmetic operators
template<class... Shapes>
class shape_tuple :
  public agency::tuple<Shapes...>,
  public arithmetic_tuple_facade<shape_tuple<Shapes...>>
{
  public:
    using agency::tuple<Shapes...>::tuple;
};

template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Shape1,
         class Shape2>
struct scoped_shape
{
  using type = decltype(
    agency::tuple_cat(
      detail::make_tuple_if_not_scoped<ExecutionCategory1>(std::declval<Shape1>()),
      detail::make_tuple_if_not_scoped<ExecutionCategory2>(std::declval<Shape2>())
    )
  );
};


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Shape1,
         class Shape2>
using scoped_shape_t = typename scoped_shape<
  ExecutionCategory1,
  ExecutionCategory2,
  Shape1,
  Shape2
>::type;


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Shape1,
         class Shape2>
__AGENCY_ANNOTATION
scoped_shape_t<ExecutionCategory1,ExecutionCategory2,Shape1,Shape2> make_scoped_shape(const Shape1& outer_shape, const Shape2& inner_shape)
{
  return agency::tuple_cat(
    detail::make_tuple_if_not_scoped<ExecutionCategory1>(outer_shape),
    detail::make_tuple_if_not_scoped<ExecutionCategory2>(inner_shape)
  );
}

} // end detail
} // end agency


namespace std
{


template<class... Shapes>
class tuple_size<agency::detail::shape_tuple<Shapes...>> : public std::tuple_size<agency::tuple<Shapes...>> {};

template<size_t i, class... Shapes>
class tuple_element<i,agency::detail::shape_tuple<Shapes...>> : public std::tuple_element<i,agency::tuple<Shapes...>> {};


} // end namespace std

