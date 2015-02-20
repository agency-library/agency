#pragma once

#include <agency/detail/config.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


template<typename T>
  struct has_nested_type_impl
{                    
  typedef char yes_type;
  typedef int  no_type;
  template<typename S> static yes_type test(typename S::type *);
  template<typename S> static no_type  test(...);
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef std::integral_constant<bool, value> type;
};


template<class T>
struct has_nested_type : has_nested_type_impl<T>::type {};


template<class Function, class... Args>
struct is_call_possible
  : has_nested_type<
      std::result_of<Function(Args...)>
    >
{
};


template<class,class = void> struct enable_if_call_possible;


template<class Result, class Function, class... Args>
struct enable_if_call_possible<Function(Args...),Result>
  : std::enable_if<
      agency::detail::is_call_possible<
        Function,Args...
      >::value,
      Result
    >
{};


} // end detail
} // end agency

