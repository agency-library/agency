#pragma once

#include <agency/detail/config.hpp>
#include <memory>
#include <utility>


namespace agency
{


__agency_exec_check_disable__
template<class Ptr,
         class = decltype(
           std::pointer_traits<Ptr>::to_address(std::declval<Ptr>())
         )
        >
__AGENCY_ANNOTATION
auto to_address(const Ptr& p) noexcept
  -> decltype(std::pointer_traits<Ptr>::to_address(p))
{
  return std::pointer_traits<Ptr>::to_address(p);
}


template<class T>
__AGENCY_ANNOTATION
constexpr T* to_address(T* ptr) noexcept
{
  return ptr;
}


} // end agency

