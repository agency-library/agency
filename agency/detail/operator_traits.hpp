#pragma once

#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace operator_traits_namespace
{


template<typename T1, typename T2> struct is_different : std::true_type {};


template<typename T1> struct is_different<T1,T1> : std::false_type {};


struct any
{
  template<typename T> any(T) {}
};


struct ignore {};


struct no {};


no operator+(any, any);
no operator+=(any, any);

no operator-(any, any);
no operator-=(any, any);

no operator*(any, any);
no operator*=(any, any);

no operator/(any, any);
no operator/=(any, any);

no operator<(any, any);
no operator<=(any, any);

no operator>(any, any);
no operator>=(any, any);


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_plus
  : is_different<
      decltype(std::declval<T1>() + std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_plus_assign
  : is_different<
      decltype(std::declval<T1&>() += std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_minus
  : is_different<
      decltype(std::declval<T1>() - std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_minus_assign
  : is_different<
      decltype(std::declval<T1&>() -= std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_multiplies
  : is_different<
      decltype(std::declval<T1>() * std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_multiplies_assign
  : is_different<
      decltype(std::declval<T1&>() *= std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_divides
  : is_different<
      decltype(std::declval<T1>() / std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_divides_assign
  : is_different<
      decltype(std::declval<T1&>() /= std::declval<T2>()),
      no
    >
{
};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_less
  : is_different<
      decltype(std::declval<T1>() < std::declval<T2>()),
      no
    >
{};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_less_equal
  : is_different<
      decltype(std::declval<T1>() <= std::declval<T2>()),
      no
    >
{};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_greater
  : is_different<
      decltype(std::declval<T1>() > std::declval<T2>()),
      no
    >
{};


template<typename T1, typename T2 = T1, typename Result = ignore>
struct has_operator_greater_equal
  : is_different<
      decltype(std::declval<T1>() >= std::declval<T2>()),
      no
    >
{};


template<typename, typename, typename Enable = void> struct operator_plus_result {};


template<typename T1, typename T2>
struct operator_plus_result<T1, T2, typename std::enable_if<has_operator_plus<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() + std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_plus_assign_result {};


template<typename T1, typename T2>
struct operator_plus_assign_result<T1,T2, typename std::enable_if<has_operator_plus_assign<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() += std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_minus_result {};


template<typename T1, typename T2>
struct operator_minus_result<T1,T2, typename std::enable_if<has_operator_minus<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() - std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_minus_assign_result {};


template<typename T1, typename T2>
struct operator_minus_assign_result<T1, T2, typename std::enable_if<has_operator_minus_assign<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() -= std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_multiplies_result {};


template<typename T1, typename T2>
struct operator_multiplies_result<T1, T2, typename std::enable_if<has_operator_multiplies<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() - std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_multiplies_assign_result {};


template<typename T1, typename T2>
struct operator_multiplies_assign_result<T1,T2, typename std::enable_if<has_operator_multiplies_assign<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() -= std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_divides_result {};


template<typename T1, typename T2>
struct operator_divides_result<T1,T2, typename std::enable_if<has_operator_divides<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() - std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_divides_assign_result {};


template<typename T1, typename T2>
struct operator_divides_assign_result<T1, T2, typename std::enable_if<has_operator_divides_assign<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() -= std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_less_result {};


template<typename T1, typename T2>
struct operator_less_result<T1,T2, typename std::enable_if<has_operator_less<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() < std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_less_equal_result {};


template<typename T1, typename T2>
struct operator_less_equal_result<T1,T2, typename std::enable_if<has_operator_less_equal<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() <= std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_greater_result {};


template<typename T1, typename T2>
struct operator_greater_result<T1,T2, typename std::enable_if<has_operator_greater<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() > std::declval<T2>());
};


template<typename, typename, typename Enable = void> struct operator_greater_equal_result {};


template<typename T1, typename T2>
struct operator_greater_equal_result<T1,T2, typename std::enable_if<has_operator_greater_equal<T1,T2>::value>::type>
{
  using type = decltype(std::declval<T1>() >= std::declval<T2>());
};


} // end operator_traits_namespace


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_plus : operator_traits_namespace::has_operator_plus<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_plus_assign : operator_traits_namespace::has_operator_plus_assign<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_minus : operator_traits_namespace::has_operator_minus<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_minus_assign : operator_traits_namespace::has_operator_minus_assign<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_multiplies : operator_traits_namespace::has_operator_multiplies<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_multiplies_assign : operator_traits_namespace::has_operator_multiplies_assign<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_divides : operator_traits_namespace::has_operator_divides<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_divides_assign : operator_traits_namespace::has_operator_divides_assign<T1,T2,Result> {};


template<typename T1, typename T2 = T1>
struct has_arithmetic_operators : std::integral_constant<
  bool,
  has_operator_plus<T1,T2,T1&>::value &&
  has_operator_plus_assign<T1,T2,T1&>::value &&
  has_operator_minus<T1,T2,T1&>::value &&
  has_operator_minus_assign<T1,T2,T1&>::value &&
  has_operator_multiplies<T1,T2,T1&>::value &&
  has_operator_multiplies_assign<T1,T2,T1&>::value &&
  has_operator_divides<T1,T2,T1&>::value &&
  has_operator_divides_assign<T1,T2,T1&>::value
>
{};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_less : operator_traits_namespace::has_operator_divides_assign<T1,T2,Result> {};

template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_less_equal : operator_traits_namespace::has_operator_divides_assign<T1,T2,Result> {};


template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_greater : operator_traits_namespace::has_operator_divides_assign<T1,T2,Result> {};

template<typename T1, typename T2 = T1, typename Result = operator_traits_namespace::ignore>
struct has_operator_greater_equal : operator_traits_namespace::has_operator_divides_assign<T1,T2,Result> {};


template<typename T1, typename T2 = T1>
struct operator_plus_result : operator_traits_namespace::operator_plus_result<T1,T2> {};


template<typename T1, typename T2 = typename std::remove_reference<T1>::type>
struct operator_plus_assign_result : operator_traits_namespace::operator_plus_assign_result<T1,T2> {};


template<typename T1, typename T2 = T1>
struct operator_minus_result : operator_traits_namespace::operator_minus_result<T1,T2> {};


template<typename T1, typename T2 = typename std::remove_reference<T1>::type>
struct operator_minus_assign_result : operator_traits_namespace::operator_minus_assign_result<T1,T2> {};


template<typename T1, typename T2 = T1>
struct operator_multiplies_result : operator_traits_namespace::operator_multiplies_result<T1,T2> {};


template<typename T1, typename T2 = typename std::remove_reference<T1>::type>
struct operator_multiplies_assign_result : operator_traits_namespace::operator_multiplies_assign_result<T1,T2> {};


template<typename T1, typename T2 = T1>
struct operator_divides_result : operator_traits_namespace::operator_divides_result<T1,T2> {};


template<typename T1, typename T2 = typename std::remove_reference<T1>::type>
struct operator_divides_assign_result : operator_traits_namespace::operator_divides_assign_result<T1,T2> {};


template<typename T1, typename T2 = T1>
struct operator_less_result : operator_traits_namespace::operator_less_result<T1,T2> {};


template<typename T1, typename T2 = T1>
struct operator_less_equal_result : operator_traits_namespace::operator_less_equal_result<T1,T2> {};


template<typename T1, typename T2 = T1>
struct operator_greater_result : operator_traits_namespace::operator_greater_result<T1,T2> {};


template<typename T1, typename T2 = T1>
struct operator_greater_equal_result : operator_traits_namespace::operator_greater_equal_result<T1,T2> {};


} // end detail
} // end agency

