#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/counted.hpp>
#include <agency/experimental/strided_view.hpp>

namespace agency
{
namespace experimental
{
namespace detail
{


template<class View>
class chunk_iterator
{
  public:
    using base_iterator_type = strided_iterator<range_iterator_t<View>>;
    using base_sentinel_type = strided_sentinel<range_sentinel_t<View>>;

    using value_type = counted_view<range_iterator_t<View>>;
    using reference = value_type;
    using difference_type = range_difference_t<View>;

    template<class Iterator,
             class Sentinel,
             class = typename std::enable_if<
               std::is_constructible<base_iterator_type, Iterator, difference_type>::value &&
               std::is_constructible<base_sentinel_type, Sentinel>::value
             >::type
            >
    __AGENCY_ANNOTATION
    chunk_iterator(Iterator iter, Sentinel end, difference_type chunk_size)
      : current_position_(iter, chunk_size),
        end_(end)
    {}

    __AGENCY_ANNOTATION
    chunk_iterator(View all, difference_type chunk_size)
      : chunk_iterator(all.begin(), all.end(), chunk_size)
    {}

    __AGENCY_ANNOTATION
    difference_type chunk_size() const
    {
      return current_position_.stride();
    }

    __AGENCY_ANNOTATION
    void operator++()
    {
      ++current_position_;
    }

    __AGENCY_ANNOTATION
    void operator+=(difference_type n)
    {
      current_position_ += n;
    }

    __AGENCY_ANNOTATION
    reference operator[](difference_type i) const
    {
      auto tmp = *this;
      tmp += i;
      return *tmp;
    }

    __AGENCY_ANNOTATION
    value_type operator*() const
    {
      auto end_of_current_chunk = base();
      ++end_of_current_chunk;

      if(end_of_current_chunk == end())
      {
        auto size_of_last_chunk = end().base() - base().base();
        return value_type(base().base(), size_of_last_chunk);
      }

      return value_type(base().base(), chunk_size());
    }

    __AGENCY_ANNOTATION
    const base_iterator_type& base() const
    {
      return current_position_;
    }

    __AGENCY_ANNOTATION
    const base_sentinel_type& end() const
    {
      return end_;
    }

  private:
    base_iterator_type current_position_;
    base_sentinel_type end_;
};


template<class View>
class chunk_sentinel
{
  public:
    using base_sentinel_type = typename chunk_iterator<View>::base_sentinel_type;

    __AGENCY_ANNOTATION
    chunk_sentinel(base_sentinel_type end)
      : end_(end)
    {}

    __AGENCY_ANNOTATION
    const base_sentinel_type& base() const
    {
      return end_;
    }

  private:
   base_sentinel_type end_;
};


template<class View>
__AGENCY_ANNOTATION
bool operator==(const chunk_iterator<View>& lhs, const chunk_sentinel<View>& rhs)
{
  return lhs.base() == rhs.base();
}


template<class View>
__AGENCY_ANNOTATION
bool operator!=(const chunk_iterator<View>& lhs, const chunk_sentinel<View>& rhs)
{
  return !(lhs == rhs);
}


template<class View>
__AGENCY_ANNOTATION
bool operator!=(const chunk_sentinel<View> &lhs, const chunk_iterator<View>& rhs)
{
  return rhs != lhs;
}


template<class View>
__AGENCY_ANNOTATION
typename chunk_iterator<View>::difference_type operator-(const chunk_sentinel<View>& lhs, const chunk_iterator<View>& rhs)
{
  return lhs.base() - rhs.base();
}


} // end detail


template<class Range>
class chunk_view
{
  public:
    __AGENCY_ANNOTATION
    chunk_view() = default;

    __AGENCY_ANNOTATION
    chunk_view(Range rng, detail::range_difference_t<Range> n)
      : begin_(agency::experimental::all(rng), n)
    {}

  private:
    using all_t = decltype(
      agency::experimental::all(std::declval<Range>())
    );

  public:
    using iterator = detail::chunk_iterator<all_t>;
    using difference_type = typename iterator::difference_type;
    using value_type = typename iterator::value_type;

    __AGENCY_ANNOTATION
    value_type operator[](difference_type i) const
    {
      return begin()[i];
    }

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return begin_;
    }

    using sentinel = detail::chunk_sentinel<all_t>;

    __AGENCY_ANNOTATION
    sentinel end() const
    {
      return sentinel(begin().end());
    }

    __AGENCY_ANNOTATION
    difference_type chunk_size() const
    {
      return begin().chunk_size();
    }

    __AGENCY_ANNOTATION
    difference_type size() const
    {
      return (end().base().base() - begin().base().base() + chunk_size() - 1) / (chunk_size());
    }

  private:
    iterator begin_;
}; // end chunk_view


template<class Range>
__AGENCY_ANNOTATION
auto chunk(Range&& rng, detail::decay_range_difference_t<Range> chunk_size) ->
  chunk_view<
    decltype(experimental::all(std::forward<Range>(rng)))
  >
{
  auto view_of_rng = experimental::all(std::forward<Range>(rng));
  return chunk_view<decltype(view_of_rng)>(view_of_rng, chunk_size);
}


} // end experimental
} // end agency

