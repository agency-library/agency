#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ndarray/detail/has_shape_member.hpp>
#include <agency/experimental/ndarray/detail/has_shape_free_function.hpp>
#include <agency/experimental/ranges/size.hpp>
#include <utility>


namespace agency
{
namespace experimental
{
namespace detail
{


// this is the type of agency::experimental::shape
struct shape_customization_point
{
  // these overloads of operator() are listed in decreasing priority order

  // member function a.shape() overload
  __agency_exec_check_disable__
  template<class A,
           __AGENCY_REQUIRES(has_shape_member<A>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(A&& a) const ->
    decltype(std::forward<A>(a).shape())
  {
    return std::forward<A>(a).shape();
  }

  // free function shape(a) overload
  __agency_exec_check_disable__
  template<class A,
           __AGENCY_REQUIRES(!has_shape_member<A>::value),
           __AGENCY_REQUIRES(has_shape_free_function<A>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(A&& a) const ->
    decltype(shape(std::forward<A>(a)))
  {
    return shape(std::forward<A>(a));
  }

  // finally, try size
  __agency_exec_check_disable__
  template<class A,
           __AGENCY_REQUIRES(!has_shape_member<A>::value),
           __AGENCY_REQUIRES(!has_shape_free_function<A>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(A&& a) const ->
    decltype(agency::experimental::size(std::forward<A>(a)))
  {
    return agency::experimental::size(std::forward<A>(a));
  }
}; // end shape_customization_point


} // end detail


namespace
{


// define the all customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& shape = agency::detail::static_const<detail::shape_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make shape a __device__ variable in __device__ code
const __device__ detail::shape_customization_point shape;
#endif


} // end anonymous namespace


} // end experimental
} // end agency

