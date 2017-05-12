#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/check_for_member_functions.hpp>
#include <climits>

namespace agency
{
namespace detail
{
namespace allocator_traits_detail
{


// we can't use std::numeric_limits::max() in __device__ code,
// so roll our own check for the maximum value of an integer type
template<class Integer>
struct max_integer_value;

template<>
struct max_integer_value<unsigned char>
{
  static constexpr unsigned char value = UCHAR_MAX;
};

template<>
struct max_integer_value<char>
{
  static constexpr char value = CHAR_MAX;
};

template<>
struct max_integer_value<signed char>
{
  static constexpr signed char value = SCHAR_MAX;
};

template<>
struct max_integer_value<int>
{
  static constexpr int value = INT_MAX;
};

template<>
struct max_integer_value<unsigned int>
{
  static constexpr unsigned int value = UINT_MAX;
};

template<>
struct max_integer_value<long int>
{
  static constexpr long int value = LONG_MAX;
};

template<>
struct max_integer_value<unsigned long int>
{
  static constexpr unsigned long int value = ULONG_MAX;
};

template<>
struct max_integer_value<long long int>
{
  static constexpr long long int value = LLONG_MAX;
};

template<>
struct max_integer_value<unsigned long long int>
{
  static constexpr unsigned long long int value = ULLONG_MAX;
};


__agency_exec_check_disable__
template<class Alloc>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_max_size<Alloc>::value,
  typename allocator_traits<Alloc>::size_type
>::type
  max_size(const Alloc& a)
{
  return a.max_size();
} // end max_size()


template<class Alloc>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_max_size<Alloc>::value,
  typename allocator_traits<Alloc>::size_type
>::type
  max_size(const Alloc&)
{
  using size_type = typename allocator_traits<Alloc>::size_type;
  using value_type = typename allocator_traits<Alloc>::value_type;

  return max_integer_value<size_type>::value / sizeof(value_type);
} // end max_size()


} // end allocator_traits_detail


template<class Alloc>
__AGENCY_ANNOTATION
typename allocator_traits<Alloc>::size_type
  allocator_traits<Alloc>
    ::max_size(const Alloc& alloc)
{
  return allocator_traits_detail::max_size(alloc);
} // end allocator_traits::max_size()


} // end detail
} // end agency

