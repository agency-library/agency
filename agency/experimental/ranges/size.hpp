#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/detail/has_size_member.hpp>
#include <agency/experimental/ranges/detail/has_size_free_function.hpp>
#include <agency/detail/static_const.hpp>


namespace agency
{
namespace experimental
{
namespace detail
{


// this is the type of agency::experimental::size
struct size_customization_point
{
  // these overloads of operator() are listed in decreasing priority order

  // member function r.size() overload
  __agency_exec_check_disable__
  template<class R,
           __AGENCY_REQUIRES(has_size_member<R>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(R&& r) const ->
    decltype(std::forward<R>(r).size())
  {
    return std::forward<R>(r).size();
  }


  // free function all(r) overload
  __agency_exec_check_disable__
  template<class R,
           __AGENCY_REQUIRES(!has_size_member<R>::value),
           __AGENCY_REQUIRES(has_size_free_function<R>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(R&& r) const ->
    decltype(size(std::forward<R>(r)))
  {
    return size(std::forward<R>(r));
  }
}; // end size_customization_point


} // end detail


namespace
{


// define the size customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& size = agency::detail::static_const<detail::size_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make all a __device__ variable in __device__ code
const __device__ detail::size_customization_point size;
#endif


} // end anonymous namespace


} // end experimental
} // end agency

