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
class iterator_range
{
  public:
    using iterator = Iterator;
    using sentinel = Sentinel;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    iterator_range(iterator begin, sentinel end)
      : begin_(begin),
        end_(end)
    {}

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    iterator_range(Range&& rng)
      : iterator_range(std::forward<Range>(rng).begin(), std::forward<Range>(rng).end())
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

// iterator_ranges are already views, so don't wrap them
template<class Iterator, class Sentinel>
__AGENCY_ANNOTATION
iterator_range<Iterator,Sentinel> all(iterator_range<Iterator,Sentinel> v)
{
  return v;
}


__agency_exec_check_disable__
template<class Range>
__AGENCY_ANNOTATION
iterator_range<range_iterator_t<Range>, range_sentinel_t<Range>>
  make_iterator_range(Range&& rng)
{
  return iterator_range<range_iterator_t<Range>, range_sentinel_t<Range>>(rng.begin(), rng.end());
}


// create a view of the given range and drop the first n elements from the view
template<class Range>
__AGENCY_ANNOTATION
iterator_range<range_iterator_t<Range>, range_sentinel_t<Range>>
  drop(Range&& rng, range_difference_t<Range> n)
{
  auto result = make_iterator_range(rng);
  result.drop(n);
  return result;
}


} // end experimental
} // end agency

