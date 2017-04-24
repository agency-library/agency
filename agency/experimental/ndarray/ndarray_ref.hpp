#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/shape.hpp>
#include <agency/detail/index_lexicographical_rank.hpp>
#include <agency/coordinate.hpp>
#include <cstddef>

namespace agency
{
namespace experimental
{


// basic_ndarray_ref is a mutable view of a multidimensional array of elements.
// The layout of the array elements is in row-major, i.e. the lexicographic order of their multidimensional indices.
//
// The dimensionality of the array is given by Shape, which is a generalized shape type.
// A type is an Shape if
//   1. The type is an integral type, or
//   2. the type is a tuple where each element is a Shape.
// XXX consider whether we actually need an Index parameter
//     do we ever use different types for Shape & Index?
// XXX The reason we take Shape & Index separately is because execution agents distinguish between
//     the type of their index and the type of the shape of their group
//     Consistency is important here, but we ought to consider whether it's actually important for agents
//     to make this distinction
template<class T, class Shape, class Index = Shape>
class basic_ndarray_ref
{
  static_assert(agency::detail::index_size<Shape>::value == agency::detail::index_size<Index>::value, "Shape rank must equal Index rank.");

  public:
    using element_type = T;
    using shape_type = Shape;
    using index_type = Index;
    using size_type = decltype(agency::detail::index_space_size(std::declval<shape_type>()));
    using reference = element_type&;
    using pointer = element_type*;

    // this iterator traverses in row-major order
    using iterator = pointer;

    __AGENCY_ANNOTATION
    basic_ndarray_ref() : basic_ndarray_ref(nullptr) {}

    __AGENCY_ANNOTATION
    explicit basic_ndarray_ref(std::nullptr_t) : basic_ndarray_ref(nullptr, shape_type{}) {}

    __AGENCY_ANNOTATION
    basic_ndarray_ref(pointer ptr, shape_type shape) : data_(ptr), shape_(shape) {}

    __AGENCY_ANNOTATION
    constexpr std::size_t rank() const
    {
      return std::tuple_size<Shape>::value;
    }

    /// \brief Returns the shape.
    /// \return The shape of this `basic_ndarray_ref`.
    __AGENCY_ANNOTATION
    shape_type shape() const
    {
      return shape_;
    }

    /// \brief Returns the total number of elements.
    /// \return A the product of the size of each dimension.
    __AGENCY_ANNOTATION
    size_type size() const
    {
      return agency::detail::index_space_size(shape());
    }

    /// \brief Returns the size of the dimension of interest.
    /// \return `shape()[dimension]`
    __AGENCY_ANNOTATION
    size_type size(const size_type& dimension) const
    {
      return shape()[dimension];
    }

    /// \brief Returns a pointer to raw data.
    /// \return The address of the first element of this `basic_ndarray_ref`.
    __AGENCY_ANNOTATION
    pointer data() const
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    reference operator[](const index_type& idx) const
    {
      auto rank = agency::detail::index_lexicographical_rank(idx, shape());
      return data_[rank];
    }

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    iterator end() const
    {
      return begin() + size();
    }

  private:
    T* data_;
    shape_type shape_;
};


// ndarray_ref is shorthand for a view of a simple n-dimensional array.
// The Rank indicates which point to use for the basic_ndarray_ref's Shape parameter
template<class T, size_t rank>
using ndarray_ref = basic_ndarray_ref<T, point<std::size_t,rank>>;


} // end experimental
} // end agency

