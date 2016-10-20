#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
#include <agency/detail/memory/allocator_traits/check_for_member_functions.hpp>
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


// construct_each algorithm:
// 1. If a.construct_each(...) is well-formed, use it. otherwise
// 2. If a.construct(...) is well-formed, use it in a for loop. otherwise
// 3. Use placement new in a for loop


__agency_exec_check_disable__
template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_construct_each<Alloc,Iterator,Iterators...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_each_impl1(Alloc& a, Iterator first, Iterator last, Iterators... iters)
{
  return a.construct_each(first, last, iters...);
} // end construct_each_impl1()


__agency_exec_check_disable__
template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_construct<Alloc,typename std::iterator_traits<Iterator>::pointer, typename std::iterator_traits<Iterators>::reference...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_each_impl2(Alloc& a, Iterator first, Iterator last, Iterators... iters)
{
  for(; first != last; ++first, swallow(++iters...))
  {
    a.construct(&*first, *iters...);
  }

  return detail::make_tuple(first,iters...);
} // end construct_each_impl2()


__agency_exec_check_disable__
template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_construct<Alloc,typename std::iterator_traits<Iterator>::pointer, typename std::iterator_traits<Iterators>::reference...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_each_impl2(Alloc& a, Iterator first, Iterator last, Iterators... iters)
{
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  for(; first != last; ++first, swallow(++iters...))
  {
    new(&*first) value_type(*iters...);
  }

  return detail::make_tuple(first,iters...);
} // end construct_each_impl2()


template<class Alloc, class Iterator, class... Iterators>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_construct_each<Alloc,Iterator,Iterators...>::value,
  detail::tuple<Iterator,Iterators...>
>::type
  construct_each_impl1(Alloc& a, Iterator first, Iterator last, Iterators... iters)
{
  return construct_each_impl2(a, first, last, iters...);
} // end construct_each_impl1()


} // end allocator_traits_detail


template<class Alloc>
  template<class Iterator, class... Iterators>
__AGENCY_ANNOTATION
detail::tuple<Iterator,Iterators...> allocator_traits<Alloc>
  ::construct_each(Alloc& alloc, Iterator first, Iterator last, Iterators... iters)
{
  return allocator_traits_detail::construct_each_impl1(alloc, first, last, iters...);
} // end allocator_traits::construct_each()


} // end detail
} // end agency


