#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <utility>
#include <iterator>

namespace agency
{
namespace experimental
{
namespace detail
{


template<class Incrementable>
class counting_iterator
{
  public:
    using value_type = Incrementable;
    using reference = value_type;
    using pointer = void;
    using difference_type = decltype(std::declval<Incrementable>() - std::declval<Incrementable>());

    // XXX this should check whether Incrementable is_integral
    //     otherwise it should defer to iterator_traits
    using iterator_category = std::random_access_iterator_tag;

    counting_iterator() = default;

    counting_iterator(const counting_iterator&) = default;

    __AGENCY_ANNOTATION
    counting_iterator(const Incrementable& value)
      : value_(value)
    {}

    // pre-increment
    __AGENCY_ANNOTATION
    counting_iterator& operator++()
    {
      ++value_;
      return *this;
    }

    // post-increment
    __AGENCY_ANNOTATION
    counting_iterator operator++(int)
    {
      counting_iterator result = *this;
      ++value_;
      return result;
    }

    // pre-decrement
    __AGENCY_ANNOTATION
    counting_iterator& operator--()
    {
      --value_;
      return *this;
    }

    // post-decrement
    __AGENCY_ANNOTATION
    counting_iterator operator--(int)
    {
      counting_iterator result = *this;
      --value_;
      return result;
    }

    // plus-assign
    __AGENCY_ANNOTATION
    counting_iterator& operator+=(difference_type n)
    {
      value_ += n;
      return *this;
    }

    // plus
    __AGENCY_ANNOTATION
    counting_iterator operator+(difference_type n) const
    {
      counting_iterator result = *this;
      result += n;
      return result;
    }

    // minus-assign
    __AGENCY_ANNOTATION
    counting_iterator& operator-=(difference_type n)
    {
      value_ -= n;
      return *this;
    }

    // minus
    __AGENCY_ANNOTATION
    counting_iterator operator-(difference_type n) const
    {
      counting_iterator result = *this;
      result -= n;
      return result;
    }

    // iterator difference
    __AGENCY_ANNOTATION
    difference_type operator-(const counting_iterator& rhs) const
    {
      return value_ - rhs.value_;
    }

    // dereference
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return value_;
    }

    // bracket
    __AGENCY_ANNOTATION
    reference operator[](difference_type i) const
    {
      auto tmp = *this;
      tmp += i;
      return *tmp;
    }

    // less
    __AGENCY_ANNOTATION
    bool operator<(const counting_iterator& other) const
    {
      return value_ < other.value_;
    }

    // less equal
    __AGENCY_ANNOTATION
    bool operator<=(const counting_iterator& other) const
    {
      return value_ <= other.value_;
    }

    // greater
    __AGENCY_ANNOTATION
    bool operator>(const counting_iterator& other) const
    {
      return value_ > other.value_;
    }

    // greater equal
    __AGENCY_ANNOTATION
    bool operator>=(const counting_iterator& other) const
    {
      return value_ >= other.value_;
    }

    // equal
    __AGENCY_ANNOTATION
    bool operator==(const counting_iterator& other) const
    {
      return value_ == other.value_;
    }

    // not equal
    __AGENCY_ANNOTATION
    bool operator!=(const counting_iterator& other) const
    {
      return value_ != other.value_;
    }

  private:
    Incrementable value_;
};


} // end detail


template<class Incrementable>
class iota_view
{
  public:
    using iterator = detail::counting_iterator<Incrementable>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;

    __AGENCY_ANNOTATION
    iota_view(Incrementable begin, Incrementable end)
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

    __AGENCY_ANNOTATION
    iota_view all() const
    {
      return *this;
    }

    __AGENCY_ANNOTATION
    difference_type size() const
    {
      return end() - begin();
    }

    __AGENCY_ANNOTATION
    bool empty() const
    {
      return size() == 0;
    }

    __AGENCY_ANNOTATION
    typename std::iterator_traits<iterator>::reference operator[](difference_type i) const
    {
      return begin()[i];
    }

  private:
    iterator begin_;
    iterator end_;
};


template<class Incrementable, class OtherIncrementable,
         __AGENCY_REQUIRES(
           std::is_convertible<
             OtherIncrementable, Incrementable
           >::value
         )>
__AGENCY_ANNOTATION
iota_view<Incrementable> iota(Incrementable begin, OtherIncrementable end)
{
  return iota_view<Incrementable>(begin, end);
}


} // end experimental
} // end agency

