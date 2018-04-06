#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/tuple/arithmetic_tuple_facade.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/make_tuple_if.hpp>


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


template<size_t OuterDepth,
         size_t InnerDepth,
         class OuterShape,
         class InnerShape>
struct scoped_shape
{
  using type = decltype(
    agency::tuple_cat(
      detail::make_tuple_if<OuterDepth == 1>(std::declval<OuterShape>()),
      detail::make_tuple_if<InnerDepth == 1>(std::declval<InnerShape>())
    )
  );
};



template<size_t OuterDepth,
         size_t InnerDepth,
         class OuterShape,
         class InnerShape>
using scoped_shape_t = typename scoped_shape<
  OuterDepth,
  InnerDepth,
  OuterShape,
  InnerShape
>::type;


template<size_t OuterDepth,
         size_t InnerDepth,
         class OuterShape,
         class InnerShape>
__AGENCY_ANNOTATION
scoped_shape_t<OuterDepth,InnerDepth,OuterShape,InnerShape> make_scoped_shape(const OuterShape& outer_shape, const InnerShape& inner_shape)
{
  return agency::tuple_cat(
    detail::make_tuple_if<OuterDepth == 1>(outer_shape),
    detail::make_tuple_if<InnerDepth == 1>(inner_shape)
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

