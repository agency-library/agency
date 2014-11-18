#pragma once

#include <type_traits>
#include <tuple>

#define __DEFINE_HAS_NESTED_TYPE(trait_name, nested_type_name) \
template<typename T> \
  struct trait_name  \
{                    \
  typedef char yes_type; \
  typedef int  no_type;  \
  template<typename S> static yes_type test(typename S::nested_type_name *); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};

namespace agency
{
namespace detail
{


template<bool b, typename T, typename F>
struct lazy_conditional
{
  using type = typename T::type;
};


template<typename T, typename F>
struct lazy_conditional<false,T,F>
{
  using type = typename F::type;
};


template<typename T>
struct identity
{
  typedef T type;
};


template<class T>
using decay_t = typename std::decay<T>::type;


template<class T>
using result_of_t = typename std::result_of<T>::type;


} // end detail
} // end agency

