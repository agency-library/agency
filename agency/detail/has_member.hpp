#pragma once

#include <agency/detail/config.hpp>

#define __DEFINE_HAS_MEMBER_TYPE(trait_name, member_type_name) \
template<typename T> \
  struct trait_name  \
{                    \
  typedef char yes_type; \
  typedef int  no_type;  \
  template<typename S> static yes_type test(typename S::member_type_name *); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};

#define __DEFINE_HAS_MEMBER_CLASS_TEMPLATE(trait_name, member_class_template_name) \
template<typename T, typename... Types> \
  struct trait_name         \
{                           \
  typedef char yes_type;    \
  typedef int  no_type;     \
  template<typename S> static yes_type test(typename S::template member_class_template_name<Types...> *); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};

#ifdef __NVCC__
#define __DEFINE_HAS_MEMBER(trait_name, member_name) \
template<typename T> \
  struct trait_name  \
{                    \
  typedef char yes_type; \
  typedef int  no_type;  \
  template<int i> struct swallow_int {}; \
  template<typename S> static yes_type test(swallow_int<sizeof(S::member_name)>*); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};
#else
#define __DEFINE_HAS_MEMBER(trait_name, member_name) \
template<typename T> \
  struct trait_name  \
{                    \
  typedef char yes_type; \
  typedef int  no_type;  \
  template<typename S> static yes_type test(decltype(S::member_name)*); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
  typedef std::integral_constant<bool, value> type;\
};
#endif

