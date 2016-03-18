#pragma once

#include <agency/detail/config.hpp>
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
    explicit span(std::nullptr_t) : span(nullptr, 0) {}

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

    // XXX should require iterator contiguity, but that requires contiguous_iterator_tag
    __agency_hd_warning_disable__
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

  private:
    pointer data_;
};


// XXX segmented_span might be a bad name because "span" probably implies contiguous
template<class T, std::ptrdiff_t NumSegments = dynamic_extent>
class segmented_span
{
  public:
    using element_type = T;
    using reference = element_type&;

    // XXX should give this thing iterators somehow

    constexpr static std::ptrdiff_t num_segments = NumSegments;
    static_assert(num_segments == 2, "segmented_span: NumSegments must be 2");

    template<class... Spans,
             class = typename std::enable_if<
               sizeof...(Spans) == num_segments
             >::type>
    __host__ __device__
    segmented_span(Spans... segments)
      : spans_{segments...}
    {}

    __host__ __device__
    reference operator[](size_t i) const
    {
      auto size0 = spans_[0].size();
      return i < size0 ? spans_[0][i] : spans_[1][i - size0];
    }

    size_t size() const
    {
      return spans_[0].size() + spans_[1].size();
    }

  private:
    span<T> spans_[num_segments];
};

// XXX implement this
template<class T>
class segmented_span<T, dynamic_extent>;


} // end experimental
} // end agency

