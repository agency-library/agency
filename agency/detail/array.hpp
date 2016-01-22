#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/swap.hpp>
#include <utility>
#include <memory>

namespace agency
{
namespace detail
{


template<class T, class Shape = size_t, class Alloc = std::allocator<T>, class Index = Shape>
class array
{
  public:
    using value_type = T;

    using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<value_type>;

    using pointer = typename std::allocator_traits<allocator_type>::pointer;

    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

    using shape_type = Shape;

    using index_type = Index;

    __AGENCY_ANNOTATION
    array() : shape_{}, data_(nullptr) {}

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    array(const shape_type& shape)
      : shape_(shape)
    {
      allocator_type alloc;
      data_ = alloc.allocate(size());
    }

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    array(const shape_type& shape, const T& val)
      : array(shape)
    {
      for(auto& x : *this)
      {
        x = val;
      }
    }

    __agency_hd_warning_disable__
    template<class Iterator>
    __AGENCY_ANNOTATION
    array(Iterator first, Iterator last)
      : array(shape_cast<shape_type>(last - first))
    {
      for(auto result = begin(); result != end(); ++result, ++first)
      {
        *result = *first;
      }
    }

    __AGENCY_ANNOTATION
    array(const array& other)
      : array(other.shape())
    {
      auto iter = other.begin();
      auto result = begin();

      for(; iter != other.end(); ++iter, ++result)
      {
        *result = *iter;
      }
    }

    __AGENCY_ANNOTATION
    array(array&& other)
      : shape_{}, data_{}
    {
      agency::detail::swap(shape_, other.shape_);
      agency::detail::swap(data_,  other.data_);
    }

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    ~array()
    {
      allocator_type alloc;
      alloc.deallocate(data_, size());
    }

    __AGENCY_ANNOTATION
    array& operator=(array&& other)
    {
      using agency::detail::swap;
      swap(shape_, other.shape_);
      swap(data_,  other.data_);

      return *this;
    }

    __AGENCY_ANNOTATION
    value_type& operator[](index_type idx)
    {
      std::size_t idx_1d = agency::detail::index_cast<std::size_t>(idx, shape(), size());

      return data_[idx_1d];
    }

    __AGENCY_ANNOTATION
    shape_type shape() const
    {
      return shape_;
    }

    __AGENCY_ANNOTATION
    std::size_t size() const
    {
      return agency::detail::shape_cast<std::size_t>(shape_);
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    const_pointer data() const
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    pointer begin()
    {
      return data();
    }

    __AGENCY_ANNOTATION
    pointer end()
    {
      return begin() + size();
    }

    __AGENCY_ANNOTATION
    const_pointer begin() const
    {
      return data();
    }

    __AGENCY_ANNOTATION
    const_pointer end() const
    {
      return begin() + size();
    }

    __agency_hd_warning_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    bool operator==(const Range& rhs) const
    {
      auto i = begin();
      auto j = rhs.begin();

      for(; i != end(); ++i, ++j)
      {
        if(*i != *j)
        {
          return false;
        }
      }

      return true;
    }

  private:
    shape_type shape_;

    pointer data_;
};


} // end detail
} // end agency

