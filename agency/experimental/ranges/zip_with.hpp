#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/tuple/tuple_utility.hpp>
#include <type_traits>
#include <iterator>

namespace agency
{
namespace experimental
{
namespace detail
{


// XXX TODO: for completeness, the no iterators case
// XXX       might wish to derive from Function to get the empty base class optimization
template<class Function, class Iterator, class... Iterators>
class zip_with_iterator
{
  private:
    using iterator_tuple_type = agency::tuple<Iterator, Iterators...>;

    __AGENCY_ANNOTATION
    zip_with_iterator(Function f, const iterator_tuple_type& iterator_tuple)
      : f_(f),
        iterator_tuple_(iterator_tuple)
    {}

    __AGENCY_ANNOTATION
    const iterator_tuple_type& iterator_tuple() const
    {
      return iterator_tuple_;
    }

  public:
    using value_type = agency::detail::result_of_t<
      Function(
        typename std::iterator_traits<Iterator>::reference,
        typename std::iterator_traits<Iterators>::reference...
      )
    >;

    using reference = value_type;

    // XXX this should actually be some sort of proxy pointer
    using pointer = void;

    using difference_type = typename std::iterator_traits<
      typename std::tuple_element<0,iterator_tuple_type>::type
    >::difference_type;

    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

    __AGENCY_ANNOTATION
    zip_with_iterator(Function f, Iterator iter, Iterators... iters)
      : zip_with_iterator(f, agency::make_tuple(iter, iters...))
    {}

    __AGENCY_ANNOTATION
    Iterator first_iterator() const
    {
      return agency::get<0>(iterator_tuple());
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

  public:
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      // this calls dereference_functor() on each element of iterator_tuple_,
      // and passes the results of these dereferences to an invocation of f_()
      return __tu::tuple_map_with_make(dereference_functor(), f_, iterator_tuple_);
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
    zip_with_iterator& operator+=(difference_type n)
    {
      __tu::tuple_for_each(add_assign_functor{n}, iterator_tuple_);
      return *this;
    }

    __AGENCY_ANNOTATION
    zip_with_iterator operator+(difference_type n) const
    {
      zip_with_iterator result = *this;
      result += n;
      return result;
    }

    __AGENCY_ANNOTATION
    reference operator[](difference_type i) const
    {
      auto tmp = *this;
      tmp += i;
      return *tmp;
    }

    // these operators are implemented as friend functions to allow interoperation with zip_with_iterators whose
    // constituent iterator types are related to ours
    // they're implemented with return type deduction to enable/disable them with SFINAE

    // equal with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator==(const zip_with_iterator& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() == rhs.first_iterator())
    {
      return lhs.first_iterator() == rhs.first_iterator();
    }

    // not equal with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator!=(const zip_with_iterator& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() != rhs.first_iterator())
    {
      return lhs.first_iterator() != rhs.first_iterator();
    }

    // less than other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator<(const zip_with_iterator& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() < rhs.first_iterator())
    {
      return lhs.first_iterator() < rhs.first_iterator();
    }

    // less than equal with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator<=(const zip_with_iterator& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() <= rhs.first_iterator())
    {
      return lhs.first_iterator() <= rhs.first_iterator();
    }

    // greater than other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator>(const zip_with_iterator& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() > rhs.first_iterator())
    {
      return lhs.first_iterator() > rhs.first_iterator();
    }

    // greater than equal with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator>=(const zip_with_iterator& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() >= rhs.first_iterator())
    {
      return lhs.first_iterator() >= rhs.first_iterator();
    }

    // minus other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator-(const zip_with_iterator& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() - rhs.first_iterator())
    {
      return lhs.first_iterator() - rhs.first_iterator();
    }

  private:
    Function f_;
    iterator_tuple_type iterator_tuple_;
};


template<class Function, class... Iterators>
__AGENCY_ANNOTATION
zip_with_iterator<Function,Iterators...> make_zip_with_iterator(Function f, Iterators... iters)
{
  return zip_with_iterator<Function,Iterators...>(f,iters...);
}


// because the only iterator we use when testing a zip_with_iterator for equality
// or arithmetic is the first iterator in the tuple, it's wasteful to use
// a full zip_with_iterator to represent the end of a zip_with_range
// So, introduce a zip_with_sentinel that only stores a single iterator
//
// XXX TODO: for completeness, the no iterators case
template<class Function, class Iterator, class... Iterators>
class zip_with_sentinel
{
  public:
    using base_iterator_type = Iterator;

    // XXX in addition to their existing restrictions,
    //     these function templates should probably also requires that Iterators... are constructible from OtherIterators...
    //     and that Function is constructible from OtherFunction

    template<class OtherIterator,
             class = typename std::enable_if<
               std::is_constructible<base_iterator_type,OtherIterator>::value
             >::type>
    __AGENCY_ANNOTATION
    zip_with_sentinel(OtherIterator end)
      : end_(end)
    {}

    template<class OtherFunction, class OtherIterator, class... OtherIterators,
             class = typename std::enable_if<
               std::is_constructible<base_iterator_type,OtherIterator>::value
             >::type>
    __AGENCY_ANNOTATION
    zip_with_sentinel(const zip_with_iterator<OtherFunction,OtherIterator,OtherIterators...>& end)
      : zip_with_sentinel(end.first_iterator())
    {}

    __AGENCY_ANNOTATION
    const base_iterator_type& base() const
    {
      return end_;
    }

    // XXX TODO: implement all relational operators as friend functions using SFINAE to enable/disable them

    // equality comparison with other zip_with_sentinel
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator==(const zip_with_sentinel& lhs, const zip_with_sentinel<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() == rhs.base())
    {
      return lhs.base() == rhs.base();
    }

    // equality comparison with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator==(const zip_with_sentinel& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() == rhs.first_iterator())
    {
      return lhs.base() == rhs.first_iterator();
    }

    // equality comparison with other zip_with_iterator (with sentinel on the right hand side)
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator==(const zip_with_iterator<OtherIterator,OtherIterators...>& lhs, const zip_with_sentinel& rhs) ->
      decltype(lhs.first_iterator() == std::declval<Iterator>())
    {
      return lhs.first_iterator() == rhs.base();
    }

    // inequality comparison with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator!=(const zip_with_sentinel& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() != rhs.first_iterator())
    {
      return lhs.base() != rhs.first_iterator();
    }

    // inequality comparison with other zip_with_iterator (with sentinel on the right hand side)
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator!=(const zip_with_iterator<OtherIterator,OtherIterators...>& lhs, const zip_with_sentinel& rhs) ->
      decltype(lhs.first_iterator() != std::declval<Iterator>())
    {
      return lhs.first_iterator() != rhs.base();
    }

    // less than with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator<(const zip_with_sentinel& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() < rhs.first_iterator())
    {
      return lhs.base() < rhs.first_iterator();
    }

    // less than equal with other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator<=(const zip_with_sentinel& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() <= rhs.first_iterator())
    {
      return lhs.base() <= rhs.first_iterator();
    }

    // minus other zip_with_iterator
    template<class OtherIterator, class... OtherIterators>
    __AGENCY_ANNOTATION
    friend auto operator-(const zip_with_sentinel& lhs, const zip_with_iterator<OtherIterator,OtherIterators...>& rhs) ->
      decltype(std::declval<Iterator>() - rhs.first_iterator())
    {
      return lhs.base() - rhs.first_iterator();
    }

  private:
    base_iterator_type end_;
};


} // end detail


template<class Function, class... Ranges>
class zip_with_view
{
  public:
    using iterator = detail::zip_with_iterator<
      Function,
      range_iterator_t<Ranges>...
    >;

    using sentinel = detail::zip_with_sentinel<
      Function,
      range_sentinel_t<Ranges>...
    >;

    zip_with_view() = default;

    zip_with_view(const zip_with_view& other) = default;

    template<class OtherRange, class... OtherRanges,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 iterator,
                 Function, range_iterator_t<OtherRange>, range_iterator_t<OtherRanges>...
               >::value &&
               std::is_constructible<
                 sentinel,
                 range_sentinel_t<OtherRange>
               >::value
             )>
    __AGENCY_ANNOTATION
    zip_with_view(Function f, OtherRange&& rng, OtherRanges&&... rngs)
      : begin_(f, rng.begin(), rngs.begin()...),
        end_(rng.end())
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
    zip_with_view all() const
    {
      return *this;
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


template<class Function, class... Ranges>
__AGENCY_ANNOTATION
zip_with_view<Function,Ranges...> zip_with(Function f, Ranges&&... ranges)
{
  return zip_with_view<Function,Ranges...>(f, std::forward<Ranges>(ranges)...);
}


} // end experimental
} // end agency


