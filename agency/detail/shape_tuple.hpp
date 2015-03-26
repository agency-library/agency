#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/make_tuple_if_not_nested.hpp>

namespace agency
{
namespace detail
{

// there's no need for a shape_tuple analogous to index_tuple yet
// but we do need a make_nested_shape function

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

