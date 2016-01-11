#pragma once

#include <agency/detail/config.hpp>

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

