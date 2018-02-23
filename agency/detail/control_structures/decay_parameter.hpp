#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/detail/control_structures/shared_parameter.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


// this type trait computes the type of the parameter passed to a user function
// given then type of parameter passed to bulk_invoke/bulk_async/etc.
// parameters are passed by value unless they are special parameters like
// shared parameters. These are passed by reference.
template<class T>
struct decay_parameter
{
  template<class U>
  struct lazy_add_lvalue_reference
  {
    using type = typename std::add_lvalue_reference<typename U::type>::type;
  };

  // first decay the parameter
  using decayed_type = typename std::decay<T>::type;

  // when passing a parameter to the user's function:
  // if the parameter is a future, then we pass a reference to its value type
  // otherwise, we pass a copy of the decayed_type
  using type = typename detail::lazy_conditional<
    is_future<decayed_type>::value,
    lazy_add_lvalue_reference<future_result<decayed_type>>,
    identity<decayed_type>
  >::type;
};

template<class T>
using decay_parameter_t = typename decay_parameter<T>::type;


template<size_t level, class Factory>
struct decay_parameter<shared_parameter<level,Factory>>
{
  // shared_parameters are passed to the user function by reference
  using type = typename shared_parameter<level,Factory>::value_type &;
};


} // end detail
} // end agency

