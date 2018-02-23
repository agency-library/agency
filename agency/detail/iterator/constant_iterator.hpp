#pragma once

#include <agency/detail/config.hpp>
#include <memory>

namespace agency
{
namespace detail
{


template<class T>
class constant_iterator
{
  public:
    using value_type = T;
    using reference = const value_type&;
    using pointer = const value_type*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    constant_iterator() = default;

    constant_iterator(const constant_iterator&) = default;

    __AGENCY_ANNOTATION
    constant_iterator(const T& value, size_t position)
      : value_(value), position_(position)
    {}

    __AGENCY_ANNOTATION
    constant_iterator(const T& value)
      : constant_iterator(value, 0)
    {}

    // dereference
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return value_;
    }

    // subscript
    __AGENCY_ANNOTATION
    reference operator[](difference_type) const
    {
      // note that there is no need to create a temporary iterator
      // e.g. tmp = *this + n
      // because the value returned by *tmp == this->value_
      return value_;
    }

    // not equal
    __AGENCY_ANNOTATION
    bool operator!=(const constant_iterator& rhs) const
    {
      return position_ != rhs.position_;
    }

    // pre-increment
    __AGENCY_ANNOTATION
    constant_iterator& operator++()
    {
      ++position_;
      return *this;
    }

    // post-increment
    __AGENCY_ANNOTATION
    constant_iterator operator++(int)
    {
      constant_iterator result = *this;
      ++position_;
      return result;
    }

    // pre-decrement
    __AGENCY_ANNOTATION
    constant_iterator& operator--()
    {
      --position_;
      return *this;
    }

    // post-decrement
    __AGENCY_ANNOTATION
    constant_iterator operator--(int)
    {
      constant_iterator result = *this;
      --position_;
      return result;
    }

    // plus-equal
    __AGENCY_ANNOTATION
    constant_iterator& operator+=(difference_type n)
    {
      position_ += n;
      return *this;
    }

    // plus
    __AGENCY_ANNOTATION
    constant_iterator operator+(difference_type n) const
    {
      constant_iterator result = *this;
      result += n;
      return result;
    }

    // difference
    __AGENCY_ANNOTATION
    difference_type operator-(const constant_iterator& rhs)
    {
      return position_ - rhs.position_;
    }

  private:
    T value_;
    size_t position_;
};


} // end detail
} // end agency

