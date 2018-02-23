#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <memory>

namespace agency
{
namespace detail
{


template<class Iterator, class Reference = typename std::iterator_traits<Iterator>::reference>
class forwarding_iterator
{
  public:
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference = Reference;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

    forwarding_iterator() = default;

    __AGENCY_ANNOTATION
    explicit forwarding_iterator(Iterator x)
      : current_(x)
    {}

    template<class U, class UReference>
    __AGENCY_ANNOTATION
    forwarding_iterator(const forwarding_iterator<U,UReference>& other)
      : current_(other.current_)
    {}

    // dereference
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return static_cast<reference>(*current_);
    }

    // subscript
    __AGENCY_ANNOTATION
    reference operator[](difference_type n) const
    {
      forwarding_iterator tmp = *this + n;
      return *tmp;
    }

    // not equal
    __AGENCY_ANNOTATION
    bool operator!=(const forwarding_iterator& rhs) const
    {
      return current_ != rhs.current_;
    }

    // pre-increment
    __AGENCY_ANNOTATION
    forwarding_iterator& operator++()
    {
      ++current_;
      return *this;
    }

    // post-increment
    __AGENCY_ANNOTATION
    forwarding_iterator operator++(int)
    {
      forwarding_iterator result = *this;
      ++current_;
      return result;
    }

    // pre-decrement
    __AGENCY_ANNOTATION
    forwarding_iterator& operator--()
    {
      --current_;
      return *this;
    }

    // post-decrement
    __AGENCY_ANNOTATION
    forwarding_iterator operator--(int)
    {
      forwarding_iterator result = *this;
      --current_;
      return result;
    }

    // plus-equal
    __AGENCY_ANNOTATION
    forwarding_iterator& operator+=(difference_type n)
    {
      current_ += n;
      return *this;
    }

    // plus
    __AGENCY_ANNOTATION
    forwarding_iterator operator+(difference_type n) const
    {
      forwarding_iterator result = *this;
      result += n;
      return result;
    }

    // difference
    __AGENCY_ANNOTATION
    difference_type operator-(const forwarding_iterator& rhs)
    {
      return current_ - rhs.current_;
    }

    __AGENCY_ANNOTATION
    Iterator base() const
    {
      return current_;
    }

  private:
    Iterator current_;
};


template<class Reference, class Iterator>
__AGENCY_ANNOTATION
forwarding_iterator<Iterator,Reference> make_forwarding_iterator(Iterator i)
{
  return forwarding_iterator<Iterator,Reference>(i);
}


} // end detail
} // end agency

