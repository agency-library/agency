#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/member_pointer_or.hpp>
#include <agency/memory/allocator/detail/allocator_traits/member_size_type_or.hpp>
#include <agency/memory/allocator/detail/allocator_traits/member_value_type_or.hpp>
#include <agency/memory/allocator/detail/allocator_traits/check_for_member_functions.hpp>


namespace agency
{
namespace detail
{


template<class Alloc, class Size, class Pointer>
struct has_allocate_member_function_impl
{
  template<class Alloc1,
           class Result = decltype(
             std::declval<Alloc1&>().allocate(std::declval<Size>())
           ),
           __AGENCY_REQUIRES(std::is_same<Result,Pointer>::value)
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Alloc>(0));
};

template<class Alloc, class Size, class Pointer>
using has_allocate_member_function = typename has_allocate_member_function_impl<Alloc,Size,Pointer>::type;


template<class Alloc, class Pointer, class Size>
struct has_deallocate_member_function_impl
{
  template<class Alloc1,
           class = decltype(
             std::declval<Alloc1&>().deallocate(std::declval<Pointer>(), std::declval<Size>())
           )
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Alloc>(0));
};

template<class Alloc, class Pointer, class Size>
using has_deallocate_member_function = typename has_deallocate_member_function_impl<Alloc,Pointer,Size>::type;


template<class T>
struct is_allocator_impl
{
  struct missing_value_type {};

  using value_type = member_value_type_or_t<T, missing_value_type>;
  using pointer = member_pointer_or_t<T, value_type*>;
  using size_type = member_size_type_or_t<T, std::size_t>;

  using type = std::integral_constant<
    bool,

    // all Allocators have to have a member type named ::value_type
    is_not_same<value_type, missing_value_type>::value and

    // if value_type is not void, then all allocators have to have the member functions .allocate() & .deallocate()
    (std::is_void<value_type>::value or (has_allocate_member_function<T,size_type,pointer>::value and has_deallocate_member_function<T,pointer,size_type>::value))
  >;
};

template<class T>
using is_allocator = typename is_allocator_impl<T>::type;


} // end detail
} // end agency

