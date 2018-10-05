#pragma once

#include <agency/detail/config.hpp>
#include <initializer_list>

namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class C>
__AGENCY_ANNOTATION
constexpr auto data(C& c) -> decltype(c.data())
{
  return c.data();
}


__agency_exec_check_disable__
template<class C>
__AGENCY_ANNOTATION
constexpr auto data(const C& c) -> decltype(c.data())
{
  return c.data();
}


template<class T, std::size_t n>
__AGENCY_ANNOTATION
constexpr T* data(T (&array)[n]) noexcept
{
  return array;
}


template<class E>
__AGENCY_ANNOTATION
constexpr const E* data(std::initializer_list<E> il) noexcept
{
  return il.begin();
}


} // end detail
} // end agency

