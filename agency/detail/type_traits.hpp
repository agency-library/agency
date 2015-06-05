#pragma once

#include <type_traits>
#include <tuple>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>

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

#define __DEFINE_HAS_NESTED_CLASS_TEMPLATE(trait_name, nested_class_template_name) \
template<typename T, typename... Types> \
  struct trait_name         \
{                           \
  typedef char yes_type;    \
  typedef int  no_type;     \
  template<typename S> static yes_type test(typename S::template nested_class_template_name<Types...> *); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};

#ifdef __NVCC__
#define __DEFINE_HAS_NESTED_MEMBER(trait_name, nested_member_name) \
template<typename T> \
  struct trait_name  \
{                    \
  typedef char yes_type; \
  typedef int  no_type;  \
  template<int i> struct swallow_int {}; \
  template<typename S> static yes_type test(swallow_int<sizeof(S::nested_member_name)>*); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};
#else
#define __DEFINE_HAS_NESTED_MEMBER(trait_name, nested_member_name) \
template<typename T> \
  struct trait_name  \
{                    \
  typedef char yes_type; \
  typedef int  no_type;  \
  template<typename S> static yes_type test(decltype(S::nested_member_name)*); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};
#endif

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


__DEFINE_HAS_NESTED_MEMBER(has_value, value);


template<class T>
struct is_tuple : has_value<std::tuple_size<T>> {};


template<class Indices, class Tuple>
struct tuple_type_list_impl;

template<size_t... Indices, class Tuple>
struct tuple_type_list_impl<index_sequence<Indices...>, Tuple>
{
  using type = type_list<
    typename std::tuple_element<Indices,Tuple>::type...
  >;
};


template<class T, class Enable = void>
struct tuple_type_list;


template<class Tuple>
struct tuple_type_list<Tuple, typename std::enable_if<is_tuple<Tuple>::value>::type>
{
  using type = typename tuple_type_list_impl<
    make_index_sequence<std::tuple_size<Tuple>::value>,
    Tuple
  >::type;
};


template<class>
struct is_empty_tuple;


template<class T>
struct is_empty_tuple_impl_impl;


template<class... Types>
struct is_empty_tuple_impl_impl<type_list<Types...>>
{
  using type = static_and<
    static_or<
      std::is_empty<Types>,
      is_empty_tuple<Types>
    >...
  >;
};


template<class T, class Enable = void>
struct is_empty_tuple_impl : std::false_type {};


template<class Tuple>
struct is_empty_tuple_impl<Tuple, typename std::enable_if<is_tuple<Tuple>::value>::type>
{
  using type = typename is_empty_tuple_impl_impl<
    typename tuple_type_list<Tuple>::type
  >::type;
};


template<class Tuple>
struct is_empty_tuple : is_empty_tuple_impl<Tuple>::type {};


template<class T>
struct lazy_add_lvalue_reference
{
  using type = typename std::add_lvalue_reference<typename T::type>::type;
};


template<class T, template<class...> class Template>
struct is_instance_of : std::false_type {};


template<class... Types, template<class...> class Template>
struct is_instance_of<Template<Types...>,Template> : std::true_type {};


} // end detail
} // end agency

