#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <type_traits>
#include <iterator>

namespace agency
{
namespace experimental
{


template<class Iterator, class Sentinel = Iterator>
class range_view
{
  public:
    using iterator = Iterator;
    using sentinel = Sentinel;

    __AGENCY_ANNOTATION
    range_view(iterator begin, sentinel end)
      : begin_(begin),
        end_(end)
    {}

    template<class Range>
    __AGENCY_ANNOTATION
    range_view(Range&& rng)
      : range_view(std::forward<Range>(rng).begin(), std::forward<Range>(rng).end())
    {}

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return begin_;
    }

    __AGENCY_ANNOTATION
    sentinel end() const
    {
      return end_;
    }

    // "drops" the first n elements of the range by advancing the begin iterator n times
    __AGENCY_ANNOTATION
    void drop(typename std::iterator_traits<iterator>::difference_type n)
    {
      begin_ += n;
    }

    __AGENCY_ANNOTATION
    typename std::iterator_traits<iterator>::reference operator[](typename std::iterator_traits<iterator>::difference_type i)
    {
      return begin()[i];
    }

    __AGENCY_ANNOTATION
    typename std::iterator_traits<iterator>::difference_type size() const
    {
      return end() - begin();
    }

  private:
    iterator begin_;
    sentinel end_;
};

// range_views are already views, so don't wrap them
template<class Iterator, class Sentinel>
__AGENCY_ANNOTATION
range_view<Iterator,Sentinel> all(range_view<Iterator,Sentinel> v)
{
  return v;
}


template<class Range>
__AGENCY_ANNOTATION
range_view<range_iterator_t<Range>, range_sentinel_t<Range>>
  make_range_view(Range&& rng)
{
  return range_view<range_iterator_t<Range>, range_sentinel_t<Range>>(rng.begin(), rng.end());
}


// create a view of the given range and drop the first n elements from the view
template<class Range>
__AGENCY_ANNOTATION
range_view<range_iterator_t<Range>, range_sentinel_t<Range>>
  drop(Range&& rng, range_difference_t<Range> n)
{
  auto result = make_range_view(rng);
  result.drop(n);
  return result;
}


} // end experimental
} // end agency

