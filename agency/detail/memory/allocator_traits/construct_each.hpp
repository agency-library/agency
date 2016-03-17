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


// construct_each algorithm:
// 1. If a.construct_each(...) is well-formed, use it. otherwise
// 2. If a.construct(...) is well-formed, use it in a for loop. otherwise
// 3. Use placement new in a for loop


__agency_hd_warning_disable__
template<class Alloc, class Iterator, class... Args>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_construct_each<Alloc,Iterator,Args...>::value,
  Iterator
>::type
  construct_each_impl1(Alloc& a, Iterator first, Iterator last, Args&&... args)
{
  return a.construct_each(first, last, std::forward<Args>(args)...);
} // end construct_each_impl1()


__agency_hd_warning_disable__
template<class Alloc, class Iterator, class... Args>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_construct<Alloc,typename std::iterator_traits<Iterator>::pointer, Args...>::value,
  Iterator
>::type
  construct_each_impl2(Alloc& a, Iterator first, Iterator last, Args&&... args)
{
  for(; first != last; ++first)
  {
    a.construct(&*first, std::forward<Args>(args)...);
  }

  return first;
} // end construct_each_impl2()


__agency_hd_warning_disable__
template<class Alloc, class Iterator, class... Args>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_construct<Alloc,typename std::iterator_traits<Iterator>::pointer, Args...>::value,
  Iterator
>::type
  construct_each_impl2(Alloc& a, Iterator first, Iterator last, Args&&... args)
{
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  for(; first != last; ++first)
  {
    new(&*first) value_type(std::forward<Args>(args)...);
  }

  return first;
} // end construct_each_impl2()


template<class Alloc, class Iterator, class... Args>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_construct_each<Alloc,Iterator,Args...>::value,
  Iterator
>::type
  construct_each_impl1(Alloc& a, Iterator first, Iterator last, Args&&... args)
{
  return construct_each_impl2(a, first, last, std::forward<Args>(args)...);
} // end construct_each_impl1()


} // end allocator_traits_detail


template<class Alloc>
  template<class Iterator, class... Args>
__AGENCY_ANNOTATION
Iterator allocator_traits<Alloc>
  ::construct_each(Alloc& alloc, Iterator first, Iterator last, Args&&... args)
{
  return allocator_traits_detail::construct_each_impl1(alloc, first, last, std::forward<Args>(args)...);
} // end allocator_traits::construct_each()


} // end detail
} // end agency


