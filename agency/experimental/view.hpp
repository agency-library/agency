#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/array.hpp>
#include <agency/experimental/span.hpp>
#include <type_traits>
#include <iterator>

namespace agency
{
namespace experimental
{


// XXX this is only valid for contiguous containers
template<class Container>
__AGENCY_ANNOTATION
span<typename Container::value_type> all(Container& c)
{
  return span<typename Container::value_type>(c);
}

// XXX this is only valid for contiguous containers
template<class Container>
__AGENCY_ANNOTATION
span<const typename Container::value_type> all(const Container& c)
{
  return span<const typename Container::value_type>(c);
}


template<class T, std::size_t N>
__AGENCY_ANNOTATION
span<T,N> all(array<T,N>& a)
{
  return span<T,N>(a);
}


template<class T, std::size_t N>
__AGENCY_ANNOTATION
span<const T,N> all(const array<T,N>& a)
{
  return span<const T,N>(a);
}


// spans are already views, so don't wrap them
// XXX maybe should put this in span.hpp
template<class ElementType, std::ptrdiff_t Extent>
__AGENCY_ANNOTATION
span<ElementType,Extent> all(span<ElementType,Extent> s)
{
  return s;
}


template<class Iterator>
class range_view
{
  public:
    using iterator = Iterator;

    __AGENCY_ANNOTATION
    range_view(iterator begin, iterator end)
      : begin_(begin),
        end_(end)
    {}

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return begin_;
    }

    __AGENCY_ANNOTATION
    iterator end() const
    {
      return end_;
    }

    // "drops" the first n elements of the range by advancing the begin iterator n times
    __AGENCY_ANNOTATION
    void drop(typename std::iterator_traits<iterator>::difference_type n)
    {
      begin_ += n;
    }

  private:
    Iterator begin_;
    Iterator end_;
};

// range_views are already views, so don't wrap them
template<class Iterator>
__AGENCY_ANNOTATION
range_view<Iterator> all(range_view<Iterator> v)
{
  return v;
}


namespace detail
{


template<class Range>
using range_iterator_t = decltype(std::declval<Range*>()->begin());


template<class Range>
using decay_range_iterator_t = range_iterator_t<typename std::decay<Range>::type>;


template<class Range>
using range_difference_t = typename std::iterator_traits<range_iterator_t<Range>>::difference_type;


} // end detail


template<class Range>
__AGENCY_ANNOTATION
range_view<detail::decay_range_iterator_t<Range>> make_range_view(Range&& rng)
{
  return range_view<detail::decay_range_iterator_t<Range>>(rng.begin(), rng.end());
}


// create a view of the given range and drop the first n elements from the view
template<class Range>
__AGENCY_ANNOTATION
range_view<detail::range_iterator_t<typename std::decay<Range>::type>>
  drop(detail::range_difference_t<typename std::decay<Range>::type> n, Range&& rng)
{
  auto result = make_range_view(rng);
  result.drop(n);
  return result;
}


} // end experimental
} // end agency

