#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/shape.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class Index>
using merge_front_index_elements_t = merge_front_shape_elements_t<Index>;


template<size_t... Indices, class Index, class Shape>
__AGENCY_ANNOTATION
merge_front_index_elements_t<Index>
  merge_front_index_elements_impl(detail::index_sequence<Indices...>, const Index& idx, const Shape& shape)
{
  return merge_front_index_elements_t<Index>{
    agency::get<0>(idx) * agency::get<1>(shape) + agency::get<1>(idx),
    agency::get<Indices+2>(idx)...
  };
} // end merge_front_index_elements_impl()


// XXX instead of receiving a shape here, it might be better to just receive the leading dimension of shape 
//     because that is the only element the implementation actually uses
template<class Index, class Shape,
         class = typename std::enable_if<
           detail::is_tuple_like<Index>::value && detail::is_tuple_like<Shape>::value
         >::type,
         class = typename std::enable_if<
           (std::tuple_size<Index>::value == std::tuple_size<Shape>::value)
         >::type>
__AGENCY_ANNOTATION
merge_front_index_elements_t<Index>
  merge_front_index_elements(const Index& idx, const Shape& shape)
{
  return detail::merge_front_index_elements_impl(detail::make_index_sequence<std::tuple_size<Index>::value - 2>(), idx, shape);
} // end merge_front_index_elements()


// a point on the number line is bounded by another if it is strictly less than the other
template<class Integral1, class Integral2,
         class = typename std::enable_if<
           std::is_integral<Integral1>::value && std::is_integral<Integral2>::value
         >::type>
__AGENCY_ANNOTATION
bool is_bounded_by(const Integral1& x, const Integral2& bound)
{
  return x < bound;
}


// a multidimensional index is bounded by another if all of its elements are
// bounded by the corresponding element of the other
// essentially we test whether x is contained within (not lying on)
// the axis-aligned bounding box from the origin to the bound
template<class Index1, class Index2,
         class = typename std::enable_if<
           is_tuple_like<Index1>::value && is_tuple_like<Index2>::value
         >::type,
         class = typename std::enable_if<
           std::tuple_size<Index1>::value == std::tuple_size<Index2>::value
         >::type>
__AGENCY_ANNOTATION
bool is_bounded_by(const Index1& x, const Index2& bound);


// terminal case: x is bounded by bound
template<size_t i, class Index1, class Index2>
__AGENCY_ANNOTATION
typename std::enable_if<
  std::tuple_size<Index1>::value <= i,
  bool
>::type
  is_bounded_by_impl(const Index1&, const Index2&)
{
  return true;
}


// recursive case: early out if x[i] is not bounded by bound[i]
template<size_t i, class Tuple1, class Tuple2>
__AGENCY_ANNOTATION
typename std::enable_if<
  i < std::tuple_size<Tuple1>::value,
  bool
>::type
  is_bounded_by_impl(const Tuple1& x, const Tuple2& bound)
{
  return detail::is_bounded_by(agency::get<i>(x), agency::get<i>(bound)) && detail::is_bounded_by_impl<i+1>(x,bound);
}


template<class Tuple1, class Tuple2,
         class EnableIf1, class EnableIf2>
__AGENCY_ANNOTATION
bool is_bounded_by(const Tuple1& x, const Tuple2& bound)
{
  return detail::is_bounded_by_impl<0>(x,bound);
}


template<size_t n, class Index>
using index_take_t = detail::decay_t<
  decltype(
    detail::unwrap_single_element_tuple_if(
      detail::tuple_take_if<n>(std::declval<Index>())
    )
  )
>;


// note that index_take() unwraps single element tuples which result from tuple_take_if
template<size_t n, class Index>
__AGENCY_ANNOTATION
index_take_t<n,Index> index_take(const Index& s)
{
  return detail::unwrap_single_element_tuple_if(detail::tuple_take_if<n>(s));
}


} // end detail
} // end agency

