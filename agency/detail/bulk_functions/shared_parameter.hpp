#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/factory.hpp>
#include <tuple>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<size_t scope, class T, class... Args>
struct shared_parameter : public call_constructor_factory<T,Args...>
{
  using call_constructor_factory<T,Args...>::call_constructor_factory;
};


template<class T> struct is_shared_parameter : std::false_type {};
template<size_t scope, class T, class... Args>
struct is_shared_parameter<shared_parameter<scope,T,Args...>> : std::true_type {};


template<class T>
struct is_shared_parameter_ref
  : std::integral_constant<
      bool,
      (std::is_reference<T>::value && is_shared_parameter<typename std::remove_reference<T>::type>::value)
    >
{};


} // end detail


template<size_t scope, class T, class... Args>
__AGENCY_ANNOTATION
detail::shared_parameter<scope,T,Args...> share_at_scope(Args&&... args)
{
  return detail::shared_parameter<scope,T,Args...>{detail::make_tuple(std::forward<Args>(args)...)};
}


template<size_t scope, class T>
__AGENCY_ANNOTATION
detail::shared_parameter<scope,T,T> share_at_scope(const T& val)
{
  return detail::shared_parameter<scope,T,T>{detail::make_tuple(val)};
}


template<class T, class... Args>
__AGENCY_ANNOTATION
auto share(Args&&... args) ->
  decltype(agency::share_at_scope<0,T>(std::forward<Args>(args)...))
{
  return agency::share_at_scope<0,T>(std::forward<Args>(args)...);
}


template<class T>
__AGENCY_ANNOTATION
auto share(const T& val) ->
  decltype(agency::share_at_scope<0>(val))
{
  return agency::share_at_scope<0>(val);
}


} // end agency

