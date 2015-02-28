#pragma once

#include <thrust/tuple.h>
#include <type_traits>
#include <agency/detail/integer_sequence.hpp>
#include <cstddef>
#include <tuple>

namespace agency
{
namespace cuda
{
namespace detail
{

using agency::detail::tuple;
using agency::detail::get;
using agency::detail::forward_as_tuple;


template<class, class> struct tuple_of_references_impl;


template<class Tuple, size_t... I>
struct tuple_of_references_impl<Tuple,agency::detail::index_sequence<I...>>
{
  using type = tuple<
    typename std::tuple_element<I,Tuple>::type&...
  >;
};


template<class Tuple>
using tuple_of_references_t =
  typename tuple_of_references_impl<
    Tuple,
    agency::detail::make_index_sequence<std::tuple_size<Tuple>::value>
  >::type;


} // end detail
} // end cuda
} // end agency


// XXX do we still need this thrust stuff?

namespace std
{


// implement the std tuple interface for thrust::tuple
// XXX we'd specialize these for cuda::detail::tuple, but we can't specialize on template using


template<class Type1, class... Types>
struct tuple_size<thrust::tuple<Type1,Types...>> : thrust::tuple_size<thrust::tuple<Type1,Types...>> {};


template<size_t i, class Type1, class... Types>
struct tuple_element<i,thrust::tuple<Type1,Types...>> : thrust::tuple_element<i,thrust::tuple<Type1,Types...>> {};


} // end std


namespace __tu
{


// tuple_traits specialization

template<class Type1, class... Types>
struct tuple_traits<thrust::tuple<Type1,Types...>>
{
  using tuple_type = thrust::tuple<Type1,Types...>;

  static const size_t size = thrust::tuple_size<tuple_type>::value;

  template<size_t i>
  using element_type = typename thrust::tuple_element<i,tuple_type>::type;

  template<size_t i>
  __AGENCY_ANNOTATION
  static element_type<i>& get(tuple_type& t)
  {
    return thrust::get<i>(t);
  } // end get()

  template<size_t i>
  __AGENCY_ANNOTATION
  static const element_type<i>& get(const tuple_type& t)
  {
    return thrust::get<i>(t);
  } // end get()
}; // end tuple_traits


} // end __tu



