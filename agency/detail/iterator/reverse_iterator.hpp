#pragma once

#include <agency/detail/config.hpp>
#include <iterator>

namespace agency
{
namespace detail
{


template<class Iterator>
class reverse_iterator
{
  public:
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference = typename std::iterator_traits<Iterator>::reference;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

    using iterator_type = Iterator;

    reverse_iterator() = default;

    __AGENCY_ANNOTATION
    explicit reverse_iterator(Iterator x)
      : current_(x)
    {}

    template<class U>
    __AGENCY_ANNOTATION
    reverse_iterator(const reverse_iterator<U>& other)
      : current_(other.base())
    {}

    template<class U>
    __AGENCY_ANNOTATION
    reverse_iterator& operator=(const reverse_iterator<U>& other)
    {
      current_ = other.base();
      return *this;
    }

    __AGENCY_ANNOTATION
    Iterator base() const
    {
      return current_;
    }

    // dereference
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      Iterator tmp = current_;
      return *--tmp;
    }

    __AGENCY_ANNOTATION
    reference operator->() const
    {
      return &operator*();
    }

    // subscript
    __AGENCY_ANNOTATION
    reference operator[](difference_type n) const
    {
      reverse_iterator tmp = *this;
      tmp += n;
      return *tmp;
    }

    // pre-increment
    __AGENCY_ANNOTATION
    reverse_iterator& operator++()
    {
      --current_;
      return *this;
    }

    // pre-decrement
    reverse_iterator& operator--()
    {
      ++current_;
      return *this;
    }

    // post-increment
    reverse_iterator operator++(int)
    {
      reverse_iterator result = *this;
      --current_;
      return result;
    }

    // post-decrement
    reverse_iterator operator--(int)
    {
      reverse_iterator result = *this;
      ++current_;
      return result;
    }

    // plus
    reverse_iterator operator+(difference_type n) const
    {
      reverse_iterator result = *this;
      result += n;
      return result;
    }

    // minus
    reverse_iterator operator-(difference_type n) const
    {
      reverse_iterator result = *this;
      result -= n;
      return result;
    }

    // plus-equal
    reverse_iterator& operator+=(difference_type n)
    {
      current_ -= n;
      return *this;
    }

    // minus-equal
    reverse_iterator& operator-=(difference_type n)
    {
      current_ += n;
      return *this;
    }

  private:
    iterator_type current_;
};


template<class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
bool operator==(const reverse_iterator<Iterator1>& lhs,
                const reverse_iterator<Iterator2>& rhs)
{
  return lhs.base() == rhs.base();
}


template<class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
bool operator!=(const reverse_iterator<Iterator1>& lhs,
                const reverse_iterator<Iterator2>& rhs)
{
  return lhs.base() != rhs.base();
}


template<class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
bool operator<(const reverse_iterator<Iterator1>& lhs,
               const reverse_iterator<Iterator2>& rhs)
{
  return lhs.base() < rhs.base();
}


template<class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
bool operator<=(const reverse_iterator<Iterator1>& lhs,
                const reverse_iterator<Iterator2>& rhs)
{
  return lhs.base() <= rhs.base();
}


template<class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
bool operator>(const reverse_iterator<Iterator1>& lhs,
               const reverse_iterator<Iterator2>& rhs)
{
  return lhs.base() > rhs.base();
}


template<class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
bool operator>=(const reverse_iterator<Iterator1>& lhs,
                const reverse_iterator<Iterator2>& rhs)
{
  return lhs.base() >= rhs.base();
}


template<class Iterator>
__AGENCY_ANNOTATION
reverse_iterator<Iterator> make_reverse_iterator(Iterator i)
{
  return reverse_iterator<Iterator>(i);
}


} // end detail
} // end agency

