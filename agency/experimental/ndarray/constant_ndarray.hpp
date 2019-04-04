#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/shape.hpp>
#include <agency/coordinate/detail/shape/shape_size.hpp>
#include <agency/detail/iterator/constant_iterator.hpp>
#include <cstddef>
#include <tuple>

namespace agency
{
namespace experimental
{


template<class T, class Shape>
class constant_ndarray
{
  public:
    using element_type = T;
    using shape_type = Shape;
    using index_type = shape_type;
    using size_type = decltype(agency::detail::index_space_size(std::declval<shape_type>()));
    using reference = T;
    using pointer = const T*;
    using iterator = agency::detail::constant_iterator<T>;

    __AGENCY_ANNOTATION
    explicit constant_ndarray(const Shape& shape = Shape(), const T& value = T())
      : shape_(shape),
        value_(value)
    {}

    __AGENCY_ANNOTATION
    const T& value() const
    {
      return value_;
    }

    __AGENCY_ANNOTATION
    constexpr std::size_t rank() const
    {
      return std::tuple_size<Shape>::value;
    }

    __AGENCY_ANNOTATION
    shape_type shape() const
    {
      return shape_;
    }

    __AGENCY_ANNOTATION
    size_type size() const
    {
      return agency::detail::index_space_size(shape());
    }

    __AGENCY_ANNOTATION
    reference operator[](const index_type&) const
    {
      return value_;
    }

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return agency::detail::constant_iterator<T>(value_);
    }

    __AGENCY_ANNOTATION
    iterator end() const
    {
      return agency::detail::constant_iterator<T>(value_, size());
    }

    __AGENCY_ANNOTATION
    constant_ndarray all() const
    {
      return *this;
    }

  private:
    shape_type shape_;
    T value_;
};


} // end experimental
} // end agency

