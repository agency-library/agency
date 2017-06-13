#pragma once

#include <agency/detail/config.hpp>
#include <agency/container/array.hpp>
#include <agency/experimental/span.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>


namespace agency
{
namespace experimental
{


// XXX this is only valid for contiguous containers
template<class Container>
__AGENCY_ANNOTATION
span<typename Container::value_type> all(Container& c)
{
  return span<typename Container::value_type>(c);
}

// XXX this is only valid for contiguous containers
template<class Container>
__AGENCY_ANNOTATION
span<const typename Container::value_type> all(const Container& c)
{
  return span<const typename Container::value_type>(c);
}


// XXX maybe should put this in array.hpp
template<class T, std::size_t N>
__AGENCY_ANNOTATION
span<T,N> all(array<T,N>& a)
{
  return span<T,N>(a);
}


// XXX maybe should put this in array.hpp
template<class T, std::size_t N>
__AGENCY_ANNOTATION
span<const T,N> all(const array<T,N>& a)
{
  return span<const T,N>(a);
}


// spans are already views, so don't wrap them
// XXX maybe should put this in span.hpp
template<class ElementType, std::ptrdiff_t Extent>
__AGENCY_ANNOTATION
span<ElementType,Extent> all(span<ElementType,Extent> s)
{
  return s;
}


// note the diliberate use of ADL when calling all() here
template<class Range>
using all_t = agency::detail::decay_t<decltype(all(std::declval<Range>()))>;


} // end experimental
} // end agency

