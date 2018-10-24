#pragma once

#include <agency/detail/config.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace adl_swap_detail
{


// this tests whether a type T is ADL-swappable
template<class T>
struct is_adl_swappable_impl
{
  template<class U,
           class = decltype(swap(std::declval<U&>(), std::declval<U&>()))
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T>
using is_adl_swappable = typename is_adl_swappable_impl<T>::type;


// this is the default implementation of swap
// which just uses std::move
__agency_exec_check_disable__
template<class T>
__AGENCY_ANNOTATION
typename std::enable_if<
  !is_adl_swappable<T>::value
>::type
  adl_swap_impl(T& a, T& b)
{
  T tmp = std::move(a);
  a = std::move(b);
  b = std::move(tmp);
}


// this is the default implementation of swap
// which calls swap through ADL
__agency_exec_check_disable__
template<class T>
__AGENCY_ANNOTATION
typename std::enable_if<
  is_adl_swappable<T>::value
>::type
  adl_swap_impl(T& a, T& b)
{
  swap(a, b);
}


} // end adl_swap_detail


// adl_swap is like std::swap except:
// * adl_swap is not ambiguous with std::swap because it has a different name and
// * adl_swap automatically calls swap() via ADL if such a call is well-formed
//   otherwise, it uses a default implementation of swap
template<class T>
__AGENCY_ANNOTATION
void adl_swap(T& a, T& b)
{
  adl_swap_detail::adl_swap_impl(a, b);
}


} // end detail
} // end agency


