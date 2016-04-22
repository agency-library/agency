#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <vector>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


__DEFINE_HAS_MEMBER_TYPE(has_execution_category, execution_category);
__DEFINE_HAS_MEMBER_TYPE(has_index_type, index_type);
__DEFINE_HAS_MEMBER_TYPE(has_shape_type, shape_type);
__DEFINE_HAS_MEMBER_CLASS_TEMPLATE(has_container_template, container);


template<class T, class Index>
struct is_container_impl
{
  // test if T is a container by trying to index it using the bracket operator
  // XXX should also check that it is constructible from Shape
  template<class T1,
           class Index1 = Index,
           class Reference = decltype(
             (*std::declval<T1*>())[std::declval<Index1>()]
           ),
           class = typename std::enable_if<
             !std::is_void<Reference>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};


template<class T, class Index>
using is_container = typename is_container_impl<T,Index>::type;


template<class T>
struct member_execution_category
{
  using type = typename T::execution_category;
};


template<class T, class Default = parallel_execution_tag>
struct member_execution_category_with_default
  : agency::detail::lazy_conditional<
      has_execution_category<T>::value,
      member_execution_category<T>,
      agency::detail::identity<Default>
    >
{};


template<class T>
struct member_index_type
{
  using type = typename T::index_type;
};


template<class T, class Default = size_t>
struct member_index_type_with_default
  : agency::detail::lazy_conditional<
      has_index_type<T>::value,
      member_index_type<T>,
      agency::detail::identity<Default>
    >
{};


template<class T>
struct member_shape_type
{
  using type = typename T::shape_type;
};


template<class T, class Default = size_t>
struct member_shape_type_with_default
  : agency::detail::lazy_conditional<
      has_shape_type<T>::value,
      member_shape_type<T>,
      agency::detail::identity<Default>
    >
{};


template<class T, class U>
struct has_future_impl
{
  template<class> static std::false_type test(...);
  template<class X> static std::true_type test(typename X::template future<U>* = 0);

  using type = decltype(test<T>(0));
};

template<class T, class U>
using has_future = typename has_future_impl<T,U>::type;


template<class Executor, class T, bool = has_future<Executor,T>::value>
struct executor_future
{
  using type = typename Executor::template future<T>;
};

template<class Executor, class T>
struct executor_future<Executor,T,false>
{
  using type = std::future<T>;
};


template<class Executor, class T>
using executor_future_t = typename executor_future<Executor,T>::type;


template<class Executor, class T>
struct executor_shared_future
{
  // get the Executor's associated future_type for use in the tests below
  using future_type = typename executor_future<Executor,T>::type;

  // deduction of the Executor's shared_future proceeds from bottom to top below:

  template<class Executor1> static auto test2(...) -> typename future_traits<future_type>::shared_future_type; // finally, defer to future_traits
  template<class Executor1> static auto test2(int) -> decltype(std::declval<Executor1>().share_future(*std::declval<future_type*>())); // next check for a nested Executor::share_future() function

  template<class Executor1> static auto test1(...) -> decltype(test2<Executor1>(0));
  template<class Executor1> static auto test1(int) -> typename Executor1::template shared_future<T>; // first check for a nested Executor::shared_future<T>

  using type = decltype(test1<Executor>(0));
};

template<class Executor, class T>
using executor_shared_future_t = typename executor_shared_future<Executor,T>::type;


template<class T, class U>
struct has_member_allocator_template_impl
{
  template<class T1,
           class U1 = U,
           class = typename T1::template allocator<U>
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T, class U>
using has_member_allocator_template = typename has_member_allocator_template_impl<T,U>::type;


template<class T, class U, bool = has_member_allocator_template<T,U>::value>
struct member_allocator
{
};

template<class T, class U>
struct member_allocator<T,U,true>
{
  using type = typename T::template allocator<U>;
};


template<class Default, class T, class U>
using member_allocator_or_t = typename lazy_conditional<
  has_member_allocator_template<T,U>::value,
  member_allocator<T,U>,
  identity<Default>
>::type;


template<class T, class U>
struct has_member_container_template_impl
{
  template<class T1,
           class U1 = U,
           class = typename T1::template container<U>
          >
  static std::true_type test(int);
           
  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T, class U>
using has_member_container_template = typename has_member_container_template_impl<T,U>::type;


template<class T, class U, bool = has_member_container_template<T,U>::value>
struct member_container
{
};

template<class T, class U>
struct member_container<T,U,true>
{
  using type = typename T::template container<U>;
};


template<class Default, class T, class U>
using member_container_or_t = typename lazy_conditional<
  has_member_container_template<T,U>::value,
  member_container<T,U>,
  identity<Default>
>::type;


template<bool condition, class Then, class Else>
struct lazy_conditional_template
{
  template<class... T>
  using type_template = typename Then::template type_template<T...>;
};

template<class Then, class Else>
struct lazy_conditional_template<false, Then, Else>
{
  template<class... T>
  using type_template = typename Else::template type_template<T...>;
};


template<template<class...> class T>
struct identity_template
{
  template<class... U>
  using type_template = T<U...>;
};


template<class Executor>
using executor_execution_category_t = typename member_execution_category_with_default<
  Executor,
  parallel_execution_tag
>::type;


template<class Executor>
struct executor_shape
{
  // deduction of the Executor's shape_type proceeds from bottom to top below:

  template<class Executor1> static auto test2(...) -> std::size_t; // finally, just use size_t
  template<class Executor1> static auto test2(int) -> decltype(std::declval<Executor1>().shape()); // next check for a nested Executor::shape() function

  template<class Executor1> static auto test1(...) -> decltype(test2<Executor1>(0));
  template<class Executor1> static auto test1(int) -> typename Executor1::shape_type; // first check for a nested Executor::shape_type
                                                                                      // XXX should insist that the result of .shape() can convert to Executor::shape_type

  using type = decltype(test1<Executor>(0));
};


template<class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;


template<class Executor>
using executor_index_t = typename member_index_type_with_default<
  Executor,
  executor_shape_t<Executor>
>::type;


template<class Executor, class T>
using executor_container_t = member_container_or_t<std::vector<T>, Executor, T>;


// XXX this should be the other way around - container<T> should depend on allocator<T>
// XXX should check the executor for the allocator
template<class Executor, class T>
using executor_allocator_t = typename executor_container_t<Executor,T>::allocator_type;


} // end executor_traits_detail
} // end detail
} // end agency

