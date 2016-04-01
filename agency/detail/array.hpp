#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/utility.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
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

    // note that array's constructors have __agency_hd_warning_disable__
    // because Alloc's constructors may not have __AGENCY_ANNOTATION

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    array() : alloc_{}, shape_{}, data_(nullptr) {}

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    explicit array(const shape_type& shape, const allocator_type& alloc = allocator_type())
      : alloc_(allocator_type()),
        shape_(shape),
        data_(allocate_and_construct_elements(alloc_, size()))
    {
    }

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    explicit array(const shape_type& shape, const T& val, const allocator_type& alloc = allocator_type())
      : alloc_(alloc),
        shape_(shape),
        data_(allocate_and_construct_elements(alloc_, size(), val))
    {
    }

    __agency_hd_warning_disable__
    template<class Iterator,
             class = typename std::enable_if<
               !std::is_convertible<Iterator,shape_type>::value
             >::type>
    __AGENCY_ANNOTATION
    array(Iterator first, Iterator last)
      : array(shape_cast<shape_type>(last - first))
    {
      for(auto result = begin(); result != end(); ++result, ++first)
      {
        *result = *first;
      }
    }

    __agency_hd_warning_disable__
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

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    array(array&& other)
      : alloc_{}, shape_{}, data_{}
    {
      swap(other);
    }

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    ~array()
    {
      clear();
    }

    __AGENCY_ANNOTATION
    array& operator=(const array& other)
    {
      // XXX this is not a very efficient implementation
      array tmp = other;
      swap(tmp);
      return *this;
    }

    __AGENCY_ANNOTATION
    array& operator=(array&& other)
    {
      swap(other);
      return *this;
    }

    __AGENCY_ANNOTATION
    void swap(array& other)
    {
      agency::detail::swap(shape_, other.shape_);
      agency::detail::swap(data_,  other.data_);
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
    const_pointer cbegin() const
    {
      return begin();
    }

    __AGENCY_ANNOTATION
    const_pointer end() const
    {
      return begin() + size();
    }

    __AGENCY_ANNOTATION
    const_pointer cend() const
    {
      return end();
    }

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    void clear()
    {
      if(size())
      {
        // XXX should really destroy through the allocator
        for(auto& x : *this)
        {
          x.~value_type();
        }

        alloc_.deallocate(data_, size());

        shape_ = shape_type{};
      }
    }

    __agency_hd_warning_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const array& lhs, const Range& rhs)
    {
      auto i = lhs.begin();
      auto j = rhs.begin();

      for(; i != lhs.end(); ++i, ++j)
      {
        if(*i != *j)
        {
          return false;
        }
      }

      return true;
    }

    __agency_hd_warning_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const Range& lhs, const array& rhs)
    {
      return rhs == lhs;
    }

  private:
    __agency_hd_warning_disable__
    template<class... Args>
    __AGENCY_ANNOTATION
    static pointer allocate_and_construct_elements(allocator_type& alloc, size_t size, Args&&... args)
    {
      pointer result = alloc.allocate(size);

      allocator_traits<allocator_type>::construct_each(alloc, result, result + size, std::forward<Args>(args)...);

      return result;
    }

    allocator_type alloc_;

    shape_type shape_;

    pointer data_;
};


} // end detail
} // end agency

