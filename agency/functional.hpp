#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/optional.hpp>
#include <tuple>
#include <utility>
#include <type_traits>

namespace agency
{


namespace detail
{


// XXX shared_parameter should really go underneath detail/bulk_functions/
template<size_t scope, class T, class... Args>
struct shared_parameter : public factory<T,Args...>
{
  using factory<T,Args...>::factory;
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


__agency_exec_check_disable__
template<class F, class... Args>
inline __AGENCY_ANNOTATION
auto invoke(F&& f, Args&&... args) -> 
  decltype(std::forward<F>(f)(std::forward<Args>(args)...))
{
  return std::forward<F>(f)(std::forward<Args>(args)...);
}


namespace detail
{


template<class Function>
struct take_first_parameter_and_invoke
{
  mutable Function f_;

  template<class Arg1, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Arg1&& arg1, Args&&...) const
    -> decltype(
         agency::invoke(f_, std::forward<Arg1>(arg1))
       )
  {
    return agency::invoke(f_, std::forward<Arg1>(arg1));
  }
};


template<class Function>
struct take_first_two_parameters_and_invoke
{
  mutable Function f_;

  template<class Arg1, class Arg2, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Arg1&& arg1, Arg2&& arg2, Args&&...) const
    -> decltype(
         agency::invoke(f_, std::forward<Arg1>(arg1), std::forward<Arg2>(arg2))
       )
  {
    return agency::invoke(f_, std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
  }
}; // end take_first_two_parameters_and_invoke


template<class Function>
struct invoke_and_return_unit
{
  mutable Function f_;

  template<class... Args>
  __AGENCY_ANNOTATION
  unit operator()(Args&&... args)
  {
    agency::invoke(f_, std::forward<Args>(args)...);
    return unit{};
  }
};


} // end detail
} // end agency

