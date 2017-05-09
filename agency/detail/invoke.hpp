#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/unit.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>

namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class F, class... Args>
inline __AGENCY_ANNOTATION
auto invoke(F&& f, Args&&... args) -> 
  decltype(std::forward<F>(f)(std::forward<Args>(args)...))
{
  return std::forward<F>(f)(std::forward<Args>(args)...);
}


template<class F, class... Args,
         __AGENCY_REQUIRES(std::is_void<result_of_t<F(Args...)>>::value)
        >
inline __AGENCY_ANNOTATION
unit invoke_and_return_unit_if_void_result(F&& f, Args&&... args)
{
  agency::detail::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  return unit();
}


template<class F, class... Args,
         __AGENCY_REQUIRES(!std::is_void<result_of_t<F(Args...)>>::value)
        >
inline __AGENCY_ANNOTATION
result_of_t<F(Args...)> invoke_and_return_unit_if_void_result(F&& f, Args&&... args)
{
  return agency::detail::invoke(std::forward<F>(f), std::forward<Args>(args)...);
}


template<class Function>
struct take_first_parameter_and_invoke
{
  mutable Function f_;

  template<class Arg1, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Arg1&& arg1, Args&&...) const
    -> decltype(
         agency::detail::invoke(f_, std::forward<Arg1>(arg1))
       )
  {
    return agency::detail::invoke(f_, std::forward<Arg1>(arg1));
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
         agency::detail::invoke(f_, std::forward<Arg1>(arg1), std::forward<Arg2>(arg2))
       )
  {
    return agency::detail::invoke(f_, std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
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
    agency::detail::invoke(f_, std::forward<Args>(args)...);
    return unit{};
  }
};


template<class Function>
struct invoke_and_return_empty
{
  mutable Function f;

  struct empty {};

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  empty operator()(const Index& idx, Args&... args) const
  {
    agency::detail::invoke(f, idx, args...);

    // return something which can be cheaply discarded
    return empty();
  }
};


} // end detail
} // end agency

