#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/detail/has_all_member.hpp>
#include <agency/experimental/ranges/detail/has_all_free_function.hpp>
#include <agency/container/array.hpp>
#include <agency/experimental/span.hpp>
#include <agency/detail/static_const.hpp>
#include <agency/experimental/ranges/iterator_range.hpp>
#include <utility>


namespace agency
{
namespace experimental
{
namespace detail
{


// this is the type of agency::experimental::all
struct all_customization_point
{
  // these overloads of operator() are listed in decreasing priority order

  // member function r.all() overload
  __agency_exec_check_disable__
  template<class R,
           __AGENCY_REQUIRES(has_all_member<R>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(R&& r) const ->
    decltype(std::forward<R>(r).all())
  {
    return std::forward<R>(r).all();
  }


  // free function all(r) overload
  __agency_exec_check_disable__
  template<class R,
           __AGENCY_REQUIRES(!has_all_member<R>::value),
           __AGENCY_REQUIRES(has_all_free_function<R>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(R&& r) const ->
    decltype(all(std::forward<R>(r)))
  {
    return all(std::forward<R>(r));
  }


  // contiguous container overload
  template<class Container,
           __AGENCY_REQUIRES(!has_all_member<Container>::value),
           __AGENCY_REQUIRES(!has_all_free_function<Container>::value),
           __AGENCY_REQUIRES(std::is_constructible<span<typename Container::value_type>, Container&>::value)
          >
  __AGENCY_ANNOTATION
  constexpr span<typename Container::value_type> operator()(Container& c) const
  {
    return {c};
  }

  // contiguous container overload
  template<class Container,
           __AGENCY_REQUIRES(!has_all_member<Container>::value),
           __AGENCY_REQUIRES(!has_all_free_function<Container>::value),
           __AGENCY_REQUIRES(std::is_constructible<span<const typename Container::value_type>, Container&>::value)
          >
  __AGENCY_ANNOTATION
  constexpr span<const typename Container::value_type> operator()(const Container& c) const
  {
    return {c};
  }


  // default: return an iterator_range
  template<class Range,
           __AGENCY_REQUIRES(!has_all_member<Range>::value),
           __AGENCY_REQUIRES(!has_all_free_function<Range>::value),
           __AGENCY_REQUIRES(!std::is_constructible<span<typename Range::value_type>, Range&>::value)
          >
  __AGENCY_ANNOTATION
  constexpr iterator_range<range_iterator_t<Range>, range_sentinel_t<Range>>
    operator()(Range& r) const
  {
    return {r.begin(), r.end()};
  }

  // default: return an iterator_range
  template<class Range,
           __AGENCY_REQUIRES(!has_all_member<const Range>::value),
           __AGENCY_REQUIRES(!has_all_free_function<const Range>::value),
           __AGENCY_REQUIRES(!std::is_constructible<span<const typename Range::value_type>, const Range&>::value)
          >
  __AGENCY_ANNOTATION
  constexpr iterator_range<range_iterator_t<const Range>, range_sentinel_t<const Range>>
    operator()(const Range& r) const
  {
    return {r.begin(), r.end()};
  }


  // XXX maybe this should be a function in array.hpp
  template<class T, std::size_t N>
  __AGENCY_ANNOTATION
  constexpr span<T,N> operator()(array<T,N>& a) const
  {
    return {a};
  }

  // XXX maybe this should be a function in array.hpp
  template<class T, std::size_t N>
  __AGENCY_ANNOTATION
  constexpr span<const T,N> operator()(const array<T,N>& a) const
  {
    return {a};
  }
}; // end all_customization_point


} // end detail


namespace
{


// define the all customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& all = agency::detail::static_const<detail::all_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make all a __device__ variable in __device__ code
const __device__ detail::all_customization_point all;
#endif


} // end anonymous namespace


template<class Range>
using all_t = agency::detail::decay_t<decltype(agency::experimental::all(std::declval<Range>()))>;


} // end experimental
} // end agency

