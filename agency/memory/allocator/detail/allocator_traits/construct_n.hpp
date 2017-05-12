#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/check_for_member_functions.hpp>
#include <memory>

namespace agency
{
namespace detail
{
namespace allocator_traits_detail
{


template<class... Args>
__AGENCY_ANNOTATION
void swallow(Args&&...)
{
}


// construct_n algorithm:
// 1. If a.construct_n(...) is well-formed, use it. otherwise
// 2. If a.construct(...) is well-formed, use it in a for loop. otherwise
// 3. Use placement new in a for loop


__agency_exec_check_disable__
template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_construct_n<Alloc,Iterator,Iterators...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_n_impl1(Alloc& a, Iterator first, size_t n, Iterators... iters)
{
  return a.construct_n(first, n, iters...);
} // end construct_n_impl1()


__agency_exec_check_disable__
template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_construct<Alloc,typename std::iterator_traits<Iterator>::pointer, typename std::iterator_traits<Iterators>::reference...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_n_impl2(Alloc& a, Iterator first, size_t n, Iterators... iters)
{
  for(size_t i = 0; i < n; ++i, ++first, allocator_traits_detail::swallow(++iters...))
  {
    a.construct(&*first, *iters...);
  }

  return detail::make_tuple(first,iters...);
} // end construct_n_impl2()


__agency_exec_check_disable__
template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_construct<Alloc,typename std::iterator_traits<Iterator>::pointer, typename std::iterator_traits<Iterators>::reference...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_n_impl2(Alloc&, Iterator first, size_t n, Iterators... iters)
{
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  for(size_t i = 0; i < n; ++i, ++first, allocator_traits_detail::swallow(++iters...))
  {
    new(&*first) value_type(*iters...);
  }

  return detail::make_tuple(first,iters...);
} // end construct_n_impl2()


template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_construct_n<Alloc,Iterator,Iterators...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_n_impl1(Alloc& a, Iterator first, size_t n, Iterators... iters)
{
  return construct_n_impl2(a, first, n, iters...);
} // end construct_n_impl1()


} // end allocator_traits_detail


template<class Alloc>
  template<class Iterator, class... Iterators>
__AGENCY_ANNOTATION
detail::tuple<Iterator,Iterators...> allocator_traits<Alloc>
  ::construct_n(Alloc& alloc, Iterator first, size_t n, Iterators... iters)
{
  return allocator_traits_detail::construct_n_impl1(alloc, first, n, iters...);
} // end allocator_traits::construct_n()


} // end detail
} // end agency


