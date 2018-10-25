#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ndarray/detail/has_domain_member.hpp>
#include <agency/experimental/ndarray/detail/has_domain_free_function.hpp>
#include <agency/experimental/ndarray/shape.hpp>
#include <agency/coordinate/lattice.hpp>
#include <utility>


namespace agency
{
namespace experimental
{
namespace detail
{


// this is the type of agency::experimental::domain
struct domain_customization_point
{
  // these overloads of operator() are listed in decreasing priority order

  // member function a.domain() overload
  __agency_exec_check_disable__
  template<class A,
           __AGENCY_REQUIRES(has_domain_member<A>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(A&& a) const ->
    decltype(std::forward<A>(a).domain())
  {
    return std::forward<A>(a).domain();
  }

  // free function domain(a) overload
  __agency_exec_check_disable__
  template<class A,
           __AGENCY_REQUIRES(!has_domain_member<A>::value),
           __AGENCY_REQUIRES(has_domain_free_function<A>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(A&& a) const ->
    decltype(domain(std::forward<A>(a)))
  {
    return domain(std::forward<A>(a));
  }

  // finally, try using shape() generically
  __agency_exec_check_disable__
  template<class A,
           __AGENCY_REQUIRES(!has_domain_member<A>::value),
           __AGENCY_REQUIRES(!has_domain_free_function<A>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(A&& a) const ->
    decltype(agency::make_lattice(agency::experimental::shape(std::forward<A>(a))))
  {
    return agency::make_lattice(agency::experimental::shape(std::forward<A>(a)));
  }
}; // end domain_customization_point


} // end detail


namespace
{


// define the domain customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& domain = agency::detail::static_const<detail::domain_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make domain a __device__ variable in __device__ code
const __device__ detail::domain_customization_point domain;
#endif


} // end anonymous namespace


} // end experimental
} // end agency

