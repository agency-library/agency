#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/all.hpp>
#include <agency/experimental/ranges/counted.hpp>
#include <agency/experimental/ranges/stride.hpp>

namespace agency
{
namespace experimental
{
namespace detail
{


template<class View, class Difference = range_difference_t<View>>
class chunk_iterator
{
  public:
    using difference_type = Difference;

    using base_iterator_type = stride_iterator<range_iterator_t<View>,Difference>;
    using base_sentinel_type = stride_sentinel<range_sentinel_t<View>>;

    using value_type = counted_view<range_iterator_t<View>,Difference>;
    using reference = value_type;

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


template<class View, class Difference>
__AGENCY_ANNOTATION
bool operator==(const chunk_iterator<View,Difference>& lhs, const chunk_sentinel<View>& rhs)
{
  return lhs.base() == rhs.base();
}


template<class View, class Difference>
__AGENCY_ANNOTATION
bool operator!=(const chunk_iterator<View,Difference>& lhs, const chunk_sentinel<View>& rhs)
{
  return !(lhs == rhs);
}


template<class View, class Difference>
__AGENCY_ANNOTATION
bool operator!=(const chunk_sentinel<View> &lhs, const chunk_iterator<View,Difference>& rhs)
{
  return rhs != lhs;
}


template<class View, class Difference>
__AGENCY_ANNOTATION
typename chunk_iterator<View,Difference>::difference_type
  operator-(const chunk_sentinel<View>& lhs, const chunk_iterator<View,Difference>& rhs)
{
  return lhs.base() - rhs.base();
}


} // end detail


template<class Range, class Difference = range_difference_t<Range>>
class chunk_view
{
  public:
    using difference_type = Difference;

    chunk_view() = default;

    __AGENCY_ANNOTATION
    chunk_view(Range rng, difference_type n)
      : begin_(all(rng), n)
    {}

  private:
    using all_t = agency::experimental::all_t<Range>;

  public:
    using iterator = detail::chunk_iterator<all_t, difference_type>;
    using value_type = typename iterator::value_type;

    __AGENCY_ANNOTATION
    value_type operator[](difference_type i) const
    {
      return begin()[i];
    }

    __AGENCY_ANNOTATION
    value_type empty_chunk() const
    {
      // XXX value_type happens to be an instantiation of counted_view
      value_type first_chunk = *begin();
      return value_type(first_chunk.begin(), difference_type(0));
    }

    __AGENCY_ANNOTATION
    value_type chunk_or_empty(difference_type i) const
    {
      return i < size() ? operator[](i) : empty_chunk();
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


template<class Range, class Difference>
__AGENCY_ANNOTATION
chunk_view<Range,Difference> chunk(Range&& rng, Difference chunk_size)
{
  return {std::forward<Range>(rng), chunk_size};
}


template<class Range, class Difference>
__AGENCY_ANNOTATION
auto chunk_evenly(Range&& rng, Difference desired_number_of_chunks) ->
  decltype(
    chunk(std::forward<Range>(rng), std::declval<Difference>())
  )
{
  // note that this calculation will not necessarily result in the desired number of chunks
  // in general, there is no way to partition a range into exactly N-1 equally-sized chunks plus a single odd-sized chunk
  Difference chunk_size = (rng.size() + desired_number_of_chunks - 1) / desired_number_of_chunks;
  return chunk(std::forward<Range>(rng), chunk_size);
}


} // end experimental
} // end agency

