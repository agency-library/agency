// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/static_const.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_has_static_property.hpp>
#include <agency/execution/executor/executor_traits/detail/has_require_member.hpp>
#include <agency/execution/executor/executor_traits/detail/has_require_free_function.hpp>
#include <agency/execution/executor/detail/adaptors/executor_ref.hpp>
#include <utility>


namespace agency
{
namespace detail
{


#if !defined(__clang__) and defined(__GNUC__) and (__GNUC__ < 6)

// gcc < 6 has trouble with agency::require
// these helpers are used to workaround the problem

template<class Executor>
struct gcc5_require_workaround_t : Executor
{
  __AGENCY_ANNOTATION
  gcc5_require_workaround_t(Executor e) : Executor{e} {}
};

template<class Executor>
__AGENCY_ANNOTATION
gcc5_require_workaround_t<decay_t<Executor>> gcc5_require_workaround(Executor&& e)
{
  return gcc5_require_workaround_t<decay_t<Executor>>{std::forward<Executor>(e)};
}

template<class T>
struct is_gcc5_require_workaround : std::false_type {};

template<class T>
struct is_gcc5_require_workaround<gcc5_require_workaround_t<T>> : std::true_type {};

#endif // __GNUC__


// this metafunction computes the result of variadic require_t::operator()
// Its purpose is to workaround (nvcc + clang)'s issues deducing
// require_t::operator()'s auto return type.
template<class E, class P, class... Ps>
struct require_result
{
  using type = typename require_result<typename require_result<E, P>::type, Ps...>::type;
};

template<class E, class P>
struct require_result<E,P>
{
  using type = conditional_t<
    executor_has_static_property<E,P>::value,
    E,
    conditional_t<
      has_require_member<E,P>::value,
      detected_t<require_member_t, E, P>,
#if !defined(__clang__) and defined(__GNUC__) and (__GNUC__ < 6)
      detected_t<require_free_function_t, gcc5_require_workaround_t<E>, P>
#else
      detected_t<require_free_function_t, E, P>
#endif // __GNUC__
    >
  >;
};

template<class E, class P, class... Ps>
using require_result_t = typename require_result<E,P,Ps...>::type;


// this is the type of agency::require
struct require_t
{
  // these overloads of operator() are listed in decreasing priority order

  template<class E, class P,
           __AGENCY_REQUIRES(
             executor_has_static_property<decay_t<E>,decay_t<P>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&&) const ->
    decltype(std::forward<E>(e))
  {
    return std::forward<E>(e);
  }


  // member function e.require(p) overload
  template<class E, class P,
           __AGENCY_REQUIRES(
             !executor_has_static_property<decay_t<E>,decay_t<P>>::value and
             has_require_member<decay_t<E>,decay_t<P>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(std::forward<E>(e).require(std::forward<P>(p)))
  {
    return std::forward<E>(e).require(std::forward<P>(p));
  }


#if !defined(__clang__) and defined(__GNUC__) and (__GNUC__ < 6)
  // free function require(e,p) overload
  template<class E, class P,
           __AGENCY_REQUIRES(
             !executor_has_static_property<decay_t<E>,decay_t<P>>::value and
             !has_require_member<decay_t<E>,decay_t<P>>::value and
             has_require_free_function<decay_t<E>,decay_t<P>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(require(gcc5_require_workaround(std::forward<E>(e)), std::forward<P>(p)))
  {
    return require(gcc5_require_workaround(std::forward<E>(e)), std::forward<P>(p));
  }
#else
  // free function require(e,p) overload
  template<class E, class P,
           __AGENCY_REQUIRES(
             !executor_has_static_property<decay_t<E>,decay_t<P>>::value and
             !has_require_member<decay_t<E>,decay_t<P>>::value and
             has_require_free_function<decay_t<E>,decay_t<P>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(require(std::forward<E>(e), std::forward<P>(p)))
  {
    return require(std::forward<E>(e), std::forward<P>(p));
  }
#endif // __GNUC__

  // variadic overload
  template<class Executor, class Property0, class Property1, class... Properties>
  __AGENCY_ANNOTATION
  constexpr require_result_t<decay_t<Executor>, decay_t<Property0>, decay_t<Property1>, decay_t<Properties>...>
    operator()(Executor&& ex, Property0&& prop0, Property1&& prop1, Properties&&... props) const
  {
    // recurse
    return operator()(operator()(std::forward<Executor>(ex), std::forward<Property0>(prop0)), std::forward<Property1>(prop1), std::forward<Properties>(props)...);
  }
};


} // end detail


namespace
{
// XXX add another nested anonymous namespace
//     to workaround nvbug 2098217
namespace
{


// define the require customization point object

#ifndef __CUDA_ARCH__
constexpr auto const& require = detail::static_const<detail::require_t>::value;
#else
// CUDA __device__ functions cannot access global variables so make require a __device__ variable in __device__ code
const __device__ detail::require_t require;
#endif


} // end anonymous namespace
} // end anonymous namespace


#if !defined(__clang__) and defined(__GNUC__) and (__GNUC__ < 6)

// g++ < 6 has problems correctly parsing calls to the require function object.
// To workaround these problems, define a function named "agency::require".
// To avoid infinite recursion between agency::require and agency::detail::require_t,
// agency::require is disabled when the incoming executor has been wrapped with the
// gcc5_workaround_t by detail::require_t.

template<class Executor, class Property, class... Properties,
         __AGENCY_REQUIRES(
           !detail::is_gcc5_require_workaround<detail::decay_t<Executor>>::value
         )>
__AGENCY_ANNOTATION
auto require(Executor&& ex, Property&& prop, Properties&&... props) ->
  decltype(detail::require_t()(std::forward<Executor>(ex), std::forward<Property>(prop), std::forward<Properties>(props)...))
{
  return detail::require_t()(std::forward<Executor>(ex), std::forward<Property>(prop), std::forward<Properties>(props)...);
}

#endif // __GNUC__


} // end agency

