#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/allocator.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/shape_cast.hpp>
#include <thrust/detail/swap.h>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T, class Shape = size_t, class Index = Shape>
class array
{
  public:
    using value_type = T;

    using pointer = value_type*;

    using const_pointer = const value_type*;

    using shape_type = Shape;

    using index_type = Index;

    __host__ __device__
    array() : shape_{}, data_(nullptr) {}

    __host__ __device__
    array(const shape_type& shape)
      : shape_(shape)
    {
      allocator<value_type> alloc;
      data_ = alloc.allocate(size());
    }

    __host__ __device__
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

    __host__ __device__
    array(array&& other)
      : shape_{}, data_{}
    {
      thrust::swap(shape_, other.shape_);
      thrust::swap(data_,  other.data_);
    }

    __host__ __device__
    ~array()
    {
      allocator<value_type> alloc;
      alloc.deallocate(data_, size());
    }

    __host__ __device__
    value_type& operator[](index_type idx)
    {
      std::size_t idx_1d = agency::detail::index_cast<std::size_t>(idx, shape(), size());

      return data_[idx_1d];
    }

    __host__ __device__
    shape_type shape() const
    {
      return shape_;
    }

    __host__ __device__
    std::size_t size() const
    {
      return agency::detail::shape_cast<std::size_t>(shape_);
    }

    __host__ __device__
    pointer data()
    {
      return data_;
    }

    __host__ __device__
    const_pointer data() const
    {
      return data_;
    }

    __host__ __device__
    pointer begin()
    {
      return data();
    }

    __host__ __device__
    pointer end()
    {
      return begin() + size();
    }

    __host__ __device__
    const_pointer begin() const
    {
      return data();
    }

    __host__ __device__
    const_pointer end() const
    {
      return begin() + size();
    }

    __agency_hd_warning_disable__
    template<class Range>
    __host__ __device__
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
} // end cuda
} // end agency

