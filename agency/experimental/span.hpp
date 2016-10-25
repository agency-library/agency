#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/array.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <cstddef>

namespace agency
{
namespace experimental
{


constexpr std::ptrdiff_t dynamic_extent = -1;


namespace detail
{


template<std::ptrdiff_t Extent>
class span_base
{
  public:
    __AGENCY_ANNOTATION
    span_base(std::ptrdiff_t)
    {
    }

    __AGENCY_ANNOTATION
    std::ptrdiff_t size() const
    {
      return Extent;
    }
};


template<>
class span_base<dynamic_extent>
{
  public:
    __AGENCY_ANNOTATION
    span_base(std::ptrdiff_t size)
      : size_(size)
    {
    }

    __AGENCY_ANNOTATION
    std::ptrdiff_t size() const
    {
      return size_;
    }

  private:
    std::ptrdiff_t size_;
};


} // end detail


// see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0122r1.pdf
template<class ElementType, std::ptrdiff_t Extent = dynamic_extent>
class span : private detail::span_base<Extent>
{
  private:
    using super_t = detail::span_base<Extent>;

  public:
    using element_type = ElementType;
    using index_type = std::ptrdiff_t;
    using pointer = element_type*;
    using reference = element_type&;
    using iterator = pointer;

    constexpr static index_type extent = Extent;

    __AGENCY_ANNOTATION
    span() : span(nullptr) {}

    __AGENCY_ANNOTATION
    explicit span(std::nullptr_t) : span(nullptr, index_type{0}) {}

    __AGENCY_ANNOTATION
    span(pointer ptr, index_type count)
      : super_t(ptr ? count : 0),
        data_(ptr)
    {}

    __AGENCY_ANNOTATION
    span(pointer first, pointer last) : span(first, last - first) {}

    template<size_t N>
    __AGENCY_ANNOTATION
    span(element_type (&arr)[N]) : span(arr, N) {}

    template<size_t N>
    __AGENCY_ANNOTATION
    span(array<typename std::remove_const<element_type>::type,N>& arr) : span(arr, N) {}

    template<size_t N>
    __AGENCY_ANNOTATION
    span(const array<typename std::remove_const<element_type>::type,N>& arr) : span(arr, N) {}

    // XXX should require iterator contiguity, but that requires contiguous_iterator_tag
    __agency_exec_check_disable__
    template<class Container,
             class BeginPointer = decltype(&*std::declval<Container>().begin()),
             class EndPointer = decltype(&*std::declval<Container>().end()),
             class = typename std::enable_if<
               std::is_convertible<BeginPointer,pointer>::value &&
               std::is_convertible<EndPointer, pointer>::value
             >::type
            >
    __AGENCY_ANNOTATION
    span(Container&& c)
      : span(&*c.begin(), &*c.end())
    {}

    __AGENCY_ANNOTATION
    index_type size() const
    {
      return super_t::size();
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    pointer data() const
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return data();
    }

    __AGENCY_ANNOTATION
    iterator end() const
    {
      return begin() + size();
    }

    __AGENCY_ANNOTATION
    reference operator[](index_type idx) const
    {
      return begin()[idx];
    }

    __AGENCY_ANNOTATION
    span<element_type, dynamic_extent> subspan(index_type offset, index_type count = dynamic_extent) const
    {
      return span<element_type, dynamic_extent>(data() + offset, count);
    }

  private:
    pointer data_;
};


template<class T, std::ptrdiff_t Extent>
__AGENCY_ANNOTATION
bool operator==(const span<T,Extent>& lhs, const span<T,Extent>& rhs)
{
  if(lhs.size() != rhs.size()) return false;

  for(auto i = 0; i < lhs.size(); ++i)
  {
    if(lhs[i] != rhs[i]) return false;
  }

  return true;
}


// specialize range_cardinality for span<T,Extent>
template<class Range>
struct range_cardinality;

template<class T>
struct range_cardinality<span<T>> : std::integral_constant<cardinality, finite> {};

template<class T, std::ptrdiff_t Extent>
struct range_cardinality<span<T,Extent>> : std::integral_constant<cardinality, static_cast<cardinality>(Extent)> {};


} // end experimental
} // end agency

