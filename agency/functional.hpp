#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/factory.hpp>
#include <type_traits>

namespace agency
{


namespace detail
{


template<size_t level, class T, class... Args>
struct shared_parameter : public factory<T,Args...>
{
  using factory<T,Args...>::factory;
};


template<class T> struct is_shared_parameter : std::false_type {};
template<size_t level, class T, class... Args>
struct is_shared_parameter<shared_parameter<level,T,Args...>> : std::true_type {};


template<class T>
struct is_shared_parameter_ref
  : std::integral_constant<
      bool,
      (std::is_reference<T>::value && is_shared_parameter<typename std::remove_reference<T>::type>::value)
    >
{};


} // end detail


template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
detail::shared_parameter<level, T,Args...> share(Args&&... args)
{
  return detail::shared_parameter<level, T,Args...>{detail::make_tuple(std::forward<Args>(args)...)};
}


template<size_t level, class T>
__AGENCY_ANNOTATION
detail::shared_parameter<level,T,T> share(const T& val)
{
  return detail::shared_parameter<level,T,T>{detail::make_tuple(val)};
}


} // end agency

