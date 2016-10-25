#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/all.hpp>
#include <type_traits>
#include <iterator>

namespace agency
{
namespace experimental
{
namespace detail
{


template<class Iterator, class Difference = typename std::iterator_traits<Iterator>::difference_type>
class stride_iterator
{
  public:
    using base_iterator_type = Iterator;
    using value_type = typename std::iterator_traits<base_iterator_type>::value_type;
    using reference = typename std::iterator_traits<base_iterator_type>::reference;
    using pointer = typename std::iterator_traits<base_iterator_type>::pointer;
    using difference_type = Difference;
    using iterator_category = typename std::iterator_traits<base_iterator_type>::iterator_category;

    template<class OtherIterator,
             class = typename std::enable_if<
               std::is_constructible<base_iterator_type,OtherIterator>::value
             >::type>
    __AGENCY_ANNOTATION
    stride_iterator(OtherIterator iter, size_t stride)
      : current_position_(iter),
        stride_(stride)
    {}

    __AGENCY_ANNOTATION
    difference_type stride() const
    {
      return stride_;
    }

    __AGENCY_ANNOTATION
    void operator++()
    {
      current_position_ += stride();
    }

    __AGENCY_ANNOTATION
    void operator+=(difference_type n)
    {
      current_position_ += n * stride();
    }

    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return *current_position_;
    }

    __AGENCY_ANNOTATION
    reference operator[](difference_type i) const
    {
      auto tmp = *this;
      tmp += i;
      return *tmp;
    }

    __AGENCY_ANNOTATION
    const base_iterator_type& base() const
    {
      return current_position_;
    }

  private:
    base_iterator_type current_position_;
    difference_type stride_;
};


template<class Iterator1, class Difference1, class Iterator2, class Difference2>
__AGENCY_ANNOTATION
auto operator==(const stride_iterator<Iterator1,Difference1>& lhs, const stride_iterator<Iterator2,Difference2>& rhs) ->
  decltype((lhs.base() == rhs.base()) && (lhs.stride() == rhs.stride()))
{
  return (lhs.base() == rhs.base()) && (lhs.stride() == rhs.stride());
}


template<class Iterator>
class stride_sentinel
{
  public:
    using base_iterator_type = Iterator;

    template<class OtherIterator,
             class = typename std::enable_if<
               std::is_constructible<base_iterator_type,OtherIterator>::value
             >::type>
    __AGENCY_ANNOTATION
    stride_sentinel(OtherIterator end)
      : end_(end)
    {}

    template<class OtherIterator, class Difference>
    __AGENCY_ANNOTATION
    auto operator==(const stride_iterator<OtherIterator,Difference>& iter) const ->
      decltype(std::declval<Iterator>() <= iter.base())
    {
      // the stride_iterator has reached the end when it is equal to or past
      // the end of the range
      return end_ <= iter.base();
    }

    __AGENCY_ANNOTATION
    const base_iterator_type& base() const
    {
      return end_;
    }

  private:
    base_iterator_type end_;
};

template<class Iterator1, class Difference, class Iterator2>
__AGENCY_ANNOTATION
auto operator==(const stride_iterator<Iterator1,Difference>& lhs, const stride_sentinel<Iterator2>& rhs) ->
  decltype(rhs == lhs)
{
  return rhs == lhs;
}


template<class Iterator1, class Difference, class Iterator2>
__AGENCY_ANNOTATION
auto operator!=(const stride_iterator<Iterator1,Difference>& lhs, const stride_sentinel<Iterator2>& rhs) ->
  decltype(!(lhs == rhs))
{
  return !(lhs == rhs);
}


template<class Iterator1, class Iterator2, class Difference>
__AGENCY_ANNOTATION
auto operator!=(const stride_sentinel<Iterator1>& lhs, const stride_iterator<Iterator2,Difference>& rhs) ->
  decltype(rhs != lhs)
{
  return rhs != lhs;
}


template<class Iterator1, class Iterator2, class Difference>
__AGENCY_ANNOTATION
Difference operator-(const stride_sentinel<Iterator1>& lhs, const stride_iterator<Iterator2,Difference>& rhs)
{
  return (lhs.base() - rhs.base() + rhs.stride() - 1) / rhs.stride();
}


} // end detail


template<class View,
         class Difference = range_difference_t<View>
        >
class stride_view
{
  private:
    using base_iterator = range_iterator_t<View>;
    using base_const_iterator = decltype(std::declval<const View*>()->begin());

    using base_sentinel = range_sentinel_t<View>;

  public:
    using iterator = detail::stride_iterator<base_iterator,Difference>;
    using index_type = Difference;
    using reference = typename std::iterator_traits<iterator>::reference;

    using sentinel = detail::stride_sentinel<base_sentinel>;

    __AGENCY_ANNOTATION
    stride_view(iterator begin, sentinel end)
      : begin_(begin), end_(end)
    {}

    __AGENCY_ANNOTATION
    stride_view(View v, index_type stride)
      : stride_view(iterator(v.begin(), stride), sentinel(v.end()))
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

    __AGENCY_ANNOTATION
    reference operator[](index_type idx) const
    {
      return begin()[idx];
    }

    __AGENCY_ANNOTATION
    bool empty() const
    {
      return begin() == end();
    }

    __AGENCY_ANNOTATION
    Difference size() const
    {
      return end() - begin();
    }

  private:
    iterator begin_;
    sentinel end_;
};


template<class Range, class Difference>
__AGENCY_ANNOTATION
stride_view<all_t<Range&&>,Difference> stride(Range&& rng, Difference stride)
{
  auto view_of_rng = all(std::forward<Range>(rng));
  return stride_view<decltype(view_of_rng), Difference>(view_of_rng, stride);
}


} // end experimental
} // end agency

