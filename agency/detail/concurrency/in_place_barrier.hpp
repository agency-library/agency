#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/variant.hpp>
#include <type_traits>
#include <cstddef>


namespace agency
{
namespace detail
{


template<class T>
struct has_index_impl
{
  template<class U, class = decltype(std::declval<U>().index())>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T>
using has_index = typename has_index_impl<T>::type;


// Some of Agency's Barrier types have constructors which take a parameter in
// addition to just a single a count parameter (e.g., as std::barrier has).
// These additional parameters may be an in_place_type_t parameter (e.g., as in
// variant_barrier) or an index parameter (e.g., as in variant_barrier, again).
// To allow components to construct both types of barriers, in_place_barrier
// wraps an underlying Barrier type in such a way that these additional
// parameters are ignored when the underlying Barrier type's constructor does
// not receive them.
//
// XXX this fancy barrier could use a better name
template<class Barrier>
struct in_place_barrier : Barrier
{
  // constructor which receives only a count parameter
  // This constructor is enabled when the underlying Barrier type has a constructor which receives only a count parameter
  template<__AGENCY_REQUIRES(
            std::is_constructible<Barrier, std::size_t>::value
          )>
  __AGENCY_ANNOTATION
  in_place_barrier(std::size_t count)
    : Barrier(count)
  {}

  // constructor which receives both an in_place_type_t and a count parameter.
  // This constructor is enabled when the underlying Barrier type has a constructor which receives both parameters.
  template<class OtherBarrier,
           __AGENCY_REQUIRES(
             std::is_constructible<Barrier, experimental::in_place_type_t<OtherBarrier>, std::size_t>::value
           )>
  __AGENCY_ANNOTATION
  in_place_barrier(experimental::in_place_type_t<OtherBarrier> which_barrier, std::size_t count)
    : Barrier(which_barrier, count)
  {}

  // constructor which receives both an index and a count parameter.
  // This constructor is enabled when the underlying Barrier type has a constructor which receives both parameters.
  template<__AGENCY_REQUIRES(
            std::is_constructible<Barrier, std::size_t, std::size_t>::value
          )>
  __AGENCY_ANNOTATION
  in_place_barrier(std::size_t index, std::size_t count)
    : Barrier(index, count)
  {}

  // constructor which receives both an in_place_type_t and a count parameter.
  // This constructor is enabled when the in_place_type_t's type parameter matches the underlying Barrier type
  // *and* the underlying Barrier type has a constructor which receives only a count parameter.
  template<__AGENCY_REQUIRES(
            std::is_constructible<Barrier, std::size_t>::value
          )>
  __AGENCY_ANNOTATION
  in_place_barrier(experimental::in_place_type_t<Barrier>, std::size_t count)
    : Barrier(count)
  {}

  // constructor which receives both an index and a count parameter.
  // This constructor is enabled when the underlying Barrier type
  //   1. has a constructor which receives only a count parameter and
  //   2. has no constructor which receives both an index and a count parameter.
  template<__AGENCY_REQUIRES(
            std::is_constructible<Barrier, std::size_t>::value and
            !std::is_constructible<Barrier, std::size_t, std::size_t>::value
          )>
  __AGENCY_ANNOTATION
  in_place_barrier(std::size_t /*index*/, std::size_t count)
    : Barrier(count)
  {}

  template<__AGENCY_REQUIRES(has_index<Barrier>::value)>
  __AGENCY_ANNOTATION
  std::size_t index() const
  {
    return Barrier::index();
  }

  template<__AGENCY_REQUIRES(!has_index<Barrier>::value)>
  __AGENCY_ANNOTATION
  std::size_t index() const
  {
    return 0;
  }
};


} // end detail
} // end agency

