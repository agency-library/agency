/// \file
/// \brief Contains definition of replace_executor.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/static_const.hpp>
#include <agency/execution/execution_policy/detail/has_replace_executor_member.hpp>
#include <agency/execution/execution_policy/detail/has_replace_executor_free_function.hpp>


namespace agency
{
namespace detail
{


// this is the type of agency::replace_executor
struct replace_executor_t
{
  // member function p.replace_executor(e) overload
  template<class P, class E,
           __AGENCY_REQUIRES(
             has_replace_executor_member<decay_t<P>,decay_t<E>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(P&& policy, E&& ex) const ->
    decltype(std::forward<P>(policy).replace_executor(std::forward<E>(ex)))
  {
    return std::forward<P>(policy).replace_executor(std::forward<E>(ex));
  }

  // free function replace_executor(p, e) overload
  template<class P, class E,
           __AGENCY_REQUIRES(
             !has_replace_executor_member<decay_t<P>,decay_t<E>>::value and
             has_replace_executor_free_function<decay_t<P>,decay_t<E>>::value
           )>
  constexpr auto operator()(P&& policy, E&& ex) const ->
    decltype(replace_executor(std::forward<P>(policy), std::forward<E>(ex)))
  {
    return replace_executor(std::forward<P>(policy), std::forward<E>(ex));
  }
};

} // end detail


namespace
{


// define the replace_executor customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& replace_executor = detail::static_const<detail::replace_executor_t>::value;
#else
// CUDA __device__ functions cannot access global variables so make replace_executor a __device__ variable in __device__ code
const __device__ detail::replace_executor_t replace_executor;
#endif

} // end anonymous namespace


} // end agency

