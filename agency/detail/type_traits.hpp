#pragma once

#include <type_traits>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>

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


template<class T, size_t n>
struct repeat_type_impl
{
  using rest = typename repeat_type_impl<T,n-1>::type;
  using type = typename type_list_prepend<
    T,
    rest
  >::type;
};

template<class T>
struct repeat_type_impl<T,0>
{
  using type = type_list<>;
};

template<class T, size_t n>
using repeat_type = typename repeat_type_impl<T,n>::type;


template<class... Conditions>
struct static_and;

template<>
struct static_and<> : std::true_type {};

template<class Condition, class... Conditions>
struct static_and<Condition, Conditions...>
  : std::integral_constant<
      bool,
      Condition::value && static_and<Conditions...>::value
    >
{};


template<class... Conditions>
struct static_or;

template<>
struct static_or<> : std::false_type {};

template<class Condition, class... Conditions>
struct static_or<Condition, Conditions...>
  : std::integral_constant<
      bool,
      Condition::value || static_or<Conditions...>::value
    >
{};


template<class T>
struct lazy_add_lvalue_reference
{
  using type = typename std::add_lvalue_reference<typename T::type>::type;
};


template<class T, template<class...> class Template>
struct is_instance_of : std::false_type {};


template<class... Types, template<class...> class Template>
struct is_instance_of<Template<Types...>,Template> : std::true_type {};


template<class Reference, class T>
struct propagate_reference;

template<class U, class T>
struct propagate_reference<U&, T>
{
  using type = T&;
};

template<class U, class T>
struct propagate_reference<const U&, T>
{
  using type = const T&;
};

template<class U, class T>
struct propagate_reference<U&&, T>
{
  using type = T&&;
};

template<class Reference, class T>
using propagate_reference_t = typename propagate_reference<Reference,T>::type;


template<class T>
struct is_not_void : std::integral_constant<bool, !std::is_void<T>::value> {};


template<class T>
struct decay_if_not_void : std::decay<T> {};

template<>
struct decay_if_not_void<void>
{
  using type = void;
};

template<class T>
using decay_if_not_void_t = typename decay_if_not_void<T>::type;


} // end detail
} // end agency

