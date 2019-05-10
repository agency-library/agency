#pragma once

#include <agency/detail/config.hpp>
#include <agency/container/array.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/detail/iterator/data.hpp>
#include <agency/experimental/ranges/size.hpp>
#include <cstddef>

namespace agency
{
namespace experimental
{


constexpr std::ptrdiff_t dynamic_extent = -1;


namespace detail
{


template<std::ptrdiff_t Extent>
class basic_span_base
{
  public:
    __AGENCY_ANNOTATION
    basic_span_base(std::ptrdiff_t)
    {
    }

    __AGENCY_ANNOTATION
    std::ptrdiff_t size() const
    {
      return Extent;
    }
};


template<>
class basic_span_base<dynamic_extent>
{
  public:
    __AGENCY_ANNOTATION
    basic_span_base(std::ptrdiff_t size)
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
template<class ElementType, std::ptrdiff_t Extent = dynamic_extent, class Pointer = ElementType*>
class basic_span : private detail::basic_span_base<Extent>
{
  private:
    static_assert(std::is_same<ElementType, typename std::pointer_traits<Pointer>::element_type>::value, "std::pointer_traits<Pointer>::element_type must be the same as ElementType.");

    using super_t = detail::basic_span_base<Extent>;

  public:
    using element_type = ElementType;
    using index_type = std::ptrdiff_t;
    using pointer = Pointer;
    using reference = typename std::iterator_traits<Pointer>::reference;
    using iterator = pointer;

    constexpr static index_type extent = Extent;

    __AGENCY_ANNOTATION
    basic_span() : basic_span(nullptr) {}

    __AGENCY_ANNOTATION
    explicit basic_span(std::nullptr_t) : basic_span(nullptr, index_type{0}) {}

    __AGENCY_ANNOTATION
    basic_span(pointer ptr, index_type count)
      : super_t(ptr ? count : 0),
        data_(ptr)
    {}

    __AGENCY_ANNOTATION
    basic_span(pointer first, pointer last) : basic_span(first, last - first) {}

    template<size_t N>
    __AGENCY_ANNOTATION
    basic_span(element_type (&arr)[N]) : basic_span(arr, N) {}

    template<size_t N>
    __AGENCY_ANNOTATION
    basic_span(array<typename std::remove_const<element_type>::type,N>& arr) : basic_span(arr, N) {}

    template<size_t N>
    __AGENCY_ANNOTATION
    basic_span(const array<typename std::remove_const<element_type>::type,N>& arr) : basic_span(arr, N) {}

    // XXX should require iterator contiguity, but that requires contiguous_iterator_tag
    __agency_exec_check_disable__
    template<class Container,
             class Data = decltype(agency::detail::data(std::declval<Container>())),
             class Size = decltype(agency::experimental::size(std::declval<Container>())),
             class = typename std::enable_if<
               std::is_constructible<basic_span,Data,Size>::value
             >::type
            >
    __AGENCY_ANNOTATION
    basic_span(Container&& c)
      : basic_span(agency::detail::data(c), agency::experimental::size(c))
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
    basic_span all() const
    {
      return *this;
    }

    __AGENCY_ANNOTATION
    reference operator[](index_type idx) const
    {
      return begin()[idx];
    }

    __AGENCY_ANNOTATION
    basic_span<element_type, dynamic_extent> subspan(index_type offset, index_type count = dynamic_extent) const
    {
      if(count == dynamic_extent) count = size() - offset;

      return basic_span<element_type, dynamic_extent>(data() + offset, count);
    }

  private:
    pointer data_;
};


template<class T, std::ptrdiff_t Extent>
__AGENCY_ANNOTATION
bool operator==(const basic_span<T,Extent>& lhs, const basic_span<T,Extent>& rhs)
{
  if(lhs.size() != rhs.size()) return false;

  for(auto i = 0; i < lhs.size(); ++i)
  {
    if(lhs[i] != rhs[i]) return false;
  }

  return true;
}


// specialize range_cardinality for basic_span<T,Extent,Pointer>
template<class Range>
struct range_cardinality;

template<class T, class Pointer>
struct range_cardinality<basic_span<T,dynamic_extent,Pointer>> : std::integral_constant<cardinality, finite> {};

template<class T, std::ptrdiff_t Extent, class Pointer>
struct range_cardinality<basic_span<T,Extent,Pointer>> : std::integral_constant<cardinality, static_cast<cardinality>(Extent)> {};


// the reason that span is a separate class and not simply an alias of basic_span
// is to avoid long type names involving basic_span<...> appearing in compiler error messages
template<class ElementType, std::ptrdiff_t Extent = dynamic_extent>
class span
  : private basic_span<ElementType, Extent>
{
  private:
    using super_t = basic_span<ElementType, Extent>;

  public:
    // types
    using element_type = typename super_t::element_type;
    using index_type = typename super_t::index_type;
    using pointer = typename super_t::pointer;
    using reference = typename super_t::reference;
    using iterator = typename super_t::iterator;

    // constructors
    using super_t::super_t;

    // dimensions
    using super_t::size;

    // raw memory access
    using super_t::data;

    // element access
    using super_t::operator[];

    // element traversal in index order
    using super_t::begin;
    using super_t::end;

    __AGENCY_ANNOTATION
    span all() const
    {
      // define a separate function instead of a using super_t::all
      // so that we can return span instead of basic_span,
      // which is the result of super_t::all()
      return *this;
    }

    __AGENCY_ANNOTATION
    span<element_type, dynamic_extent> subspan(index_type offset, index_type count = dynamic_extent) const
    {
      // define a separate function instead of using super_t::subspan
      // so that we can return span instead of basic_span,
      // which is the result of super_t::subspan()
      if(count == dynamic_extent) count = size() - offset;

      return span<element_type, dynamic_extent>(data() + offset, count);
    }
};


// specialize range_cardinality for span<T,Pointer>
template<class Range>
struct range_cardinality;

template<class T, std::ptrdiff_t Extent>
struct range_cardinality<span<T,Extent>> : range_cardinality<basic_span<T,Extent>> {};


} // end experimental
} // end agency

