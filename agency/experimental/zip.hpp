#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/view.hpp>
#include <agency/detail/tuple.hpp>
#include <type_traits>
#include <iterator>

namespace agency
{
namespace experimental
{
namespace detail
{


template<class Iterator, class... Iterators>
class zip_iterator
{
  public:
    using iterator_tuple_type = agency::detail::tuple<Iterator,Iterators...>;

    using value_type = agency::detail::tuple<
      typename std::iterator_traits<
        Iterator
      >::value_type,
      typename std::iterator_traits<
        Iterators
      >::value_type...
    >;

    using reference = agency::detail::tuple<
      typename std::iterator_traits<
        Iterator
      >::reference,
      typename std::iterator_traits<
        Iterators
      >::reference...
    >;

    // XXX this should actually be some sort of proxy pointer
    using pointer = void;

    using difference_type = typename std::iterator_traits<
      typename std::tuple_element<0,iterator_tuple_type>::type
    >::difference_type;

    // XXX this should be the common category among the iterator types
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

    __AGENCY_ANNOTATION
    zip_iterator(const iterator_tuple_type& iterator_tuple)
      : iterator_tuple_(iterator_tuple)
    {}

    __AGENCY_ANNOTATION
    zip_iterator(Iterator iter, Iterators... iters)
      : zip_iterator(agency::detail::make_tuple(iter, iters...))
    {}

    __AGENCY_ANNOTATION
    const iterator_tuple_type& iterator_tuple() const
    {
      return iterator_tuple_;
    }

  private:
    struct increment_functor
    {
      template<class OtherIterator>
      __AGENCY_ANNOTATION
      void operator()(OtherIterator& iter)
      {
        ++iter;
      }
    };

  public:
    __AGENCY_ANNOTATION
    void operator++()
    {
      __tu::tuple_for_each(increment_functor(), iterator_tuple_);
    }

  private:
    struct dereference_functor
    {
      template<class OtherIterator>
      __AGENCY_ANNOTATION
      typename std::iterator_traits<OtherIterator>::reference
        operator()(OtherIterator iter)
      {
        return *iter;
      }
    };

    struct forward_as_tuple_functor
    {
      template<class... Args>
      __AGENCY_ANNOTATION
      agency::detail::tuple<Args&&...> operator()(Args&&... args) const
      {
        return agency::detail::forward_as_tuple(args...);
      }
    };

  public:
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return __tu::tuple_map_with_make(dereference_functor(), forward_as_tuple_functor(), iterator_tuple_);
    }

  private:
    struct add_assign_functor
    {
      difference_type rhs;

      template<class OtherIterator>
      __AGENCY_ANNOTATION
      void operator()(OtherIterator& iter) const
      {
        iter += rhs;
      }
    };

  public:
    __AGENCY_ANNOTATION
    zip_iterator operator+=(difference_type n)
    {
      __tu::tuple_for_each(add_assign_functor{n}, iterator_tuple_);
      return *this;
    }

    __AGENCY_ANNOTATION
    reference operator[](difference_type i) const
    {
      auto tmp = *this;
      tmp += i;
      return *tmp;
    }

    __AGENCY_ANNOTATION
    bool operator==(const zip_iterator& rhs) const
    {
      return iterator_tuple() == rhs.iterator_tuple();
    }

    __AGENCY_ANNOTATION
    bool operator!=(const zip_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    difference_type operator-(const zip_iterator<OtherIterator,OtherIterators...>& rhs) const
    {
      return agency::detail::get<0>(iterator_tuple()) - agency::detail::get<0>(rhs.iterator_tuple());
    }

  private:
    iterator_tuple_type iterator_tuple_;
};


} // end detail


template<class Range, class... Ranges>
class zip_view
{
  public:
    using iterator = detail::zip_iterator<
      detail::decay_range_iterator_t<Range>,
      detail::decay_range_iterator_t<Ranges>...
    >;

    using sentinel = detail::zip_iterator<
      detail::decay_range_sentinel_t<Range>,
      detail::decay_range_sentinel_t<Ranges>...
    >;

    template<class OtherRange, class... OtherRanges>
    __AGENCY_ANNOTATION
    zip_view(OtherRange&& rng, OtherRanges&&... rngs)
      : begin_(rng.begin(), rngs.begin()...),
        end_(rng.end(), rngs.end()...)
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
    typename iterator::reference
      operator[](typename iterator::difference_type i)
    {
      return begin()[i];
    }

    __AGENCY_ANNOTATION
    typename iterator::difference_type
      size() const
    {
      return end() - begin();
    }

  private:
    iterator begin_;
    sentinel end_;
};


template<class Range, class... Ranges>
__AGENCY_ANNOTATION
zip_view<Range,Ranges...> zip(Range&& rng, Ranges&&... ranges)
{
  return zip_view<Range,Ranges...>(std::forward<Range>(rng), std::forward<Ranges>(ranges)...);
}


} // end experimental
} // end agency

