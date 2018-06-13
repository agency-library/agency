#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <iterator>

namespace agency
{
namespace experimental
{


template<class Iterator, class Difference = typename std::iterator_traits<Iterator>::difference_type>
class counted_view
{
  public:
    using iterator = Iterator;
    using difference_type = Difference;

    __AGENCY_ANNOTATION
    counted_view(iterator iter, difference_type n)
      : begin_(iter),
        size_(n)
    {}

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return begin_;
    }

    __AGENCY_ANNOTATION
    iterator end() const
    {
      return begin() + size();
    }

    __AGENCY_ANNOTATION
    counted_view all() const
    {
      return *this;
    }

    __AGENCY_ANNOTATION
    difference_type size() const
    {
      return size_;
    }

    __AGENCY_ANNOTATION
    bool empty() const
    {
      return size_ == 0;
    }

    __AGENCY_ANNOTATION
    typename std::iterator_traits<iterator>::reference operator[](difference_type i) const
    {
      return begin()[i];
    }

  private:
    iterator begin_;
    difference_type size_;
};


template<class Range>
__AGENCY_ANNOTATION
counted_view<range_iterator_t<Range>,range_difference_t<Range>>
  counted(Range&& rng, range_difference_t<Range> n)
{
  return counted_view<range_iterator_t<Range>,range_difference_t<Range>>(rng.begin(), n);
}

template<class Difference, class Range>
__AGENCY_ANNOTATION
counted_view<range_iterator_t<Range>,Difference>
  counted(Range&& rng, Difference n)
{
  return counted_view<range_iterator_t<Range>,Difference>(rng.begin(), n);
}


template<class Difference, class Range>
__AGENCY_ANNOTATION
counted_view<range_iterator_t<Range>,Difference>
  counted(Range&& rng, range_difference_t<Range> from, Difference n)
{
  return counted_view<range_iterator_t<Range>,Difference>(rng.begin() + from, n);
}


} // end experimental
} // end agency

