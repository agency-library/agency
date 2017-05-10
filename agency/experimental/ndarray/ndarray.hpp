#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ndarray/ndarray_ref.hpp>
#include <agency/detail/utility.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
#include <agency/detail/iterator/constant_iterator.hpp>
#include <utility>
#include <memory>
#include <iterator>


namespace agency
{
namespace experimental
{


template<class T, class Shape = size_t, class Alloc = std::allocator<T>, class Index = Shape>
class basic_ndarray
{
  private:
    using all_t = experimental::basic_ndarray_ref<T,Shape,Index>;

  public:
    using value_type = T;

    using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<value_type>;

    using pointer = typename std::allocator_traits<allocator_type>::pointer;

    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

    using shape_type = typename all_t::shape_type;

    using index_type = typename all_t::index_type;

    // note that basic_ndarray's constructors have __agency_exec_check_disable__
    // because Alloc's constructors may not have __AGENCY_ANNOTATION

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray() : alloc_{}, all_{} {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit basic_ndarray(const shape_type& shape, const allocator_type& alloc = allocator_type())
      : alloc_(alloc),
        all_(allocate_and_construct_elements(alloc_, agency::detail::shape_size(shape)), shape)
    {
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit basic_ndarray(const shape_type& shape, const T& val, const allocator_type& alloc = allocator_type())
      : alloc_(alloc),
        all_(allocate_and_construct_elements(alloc_, agency::detail::shape_size(shape), val), shape)
    {
    }

    __agency_exec_check_disable__
    template<class Iterator,
             // XXX this requirement should really be something like is_input_iterator<InputIterator>
             __AGENCY_REQUIRES(
               std::is_convertible<typename std::iterator_traits<Iterator>::value_type, value_type>::value
             )>
    __AGENCY_ANNOTATION
    basic_ndarray(Iterator first, Iterator last)
      : basic_ndarray(agency::detail::shape_cast<shape_type>(last - first))
    {
      for(auto result = begin(); result != end(); ++result, ++first)
      {
        *result = *first;
      }
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray(const basic_ndarray& other)
      : basic_ndarray(other.shape())
    {
      auto iter = other.begin();
      auto result = begin();

      for(; iter != other.end(); ++iter, ++result)
      {
        *result = *iter;
      }
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray(basic_ndarray&& other)
      : alloc_{}, all_{}
    {
      swap(other);
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    ~basic_ndarray()
    {
      clear();
    }

    __AGENCY_ANNOTATION
    basic_ndarray& operator=(const basic_ndarray& other)
    {
      // XXX this is not a very efficient implementation
      basic_ndarray tmp = other;
      swap(tmp);
      return *this;
    }

    __AGENCY_ANNOTATION
    basic_ndarray& operator=(basic_ndarray&& other)
    {
      swap(other);
      return *this;
    }

    __AGENCY_ANNOTATION
    void swap(basic_ndarray& other)
    {
      agency::detail::adl_swap(all_, other.all_);
    }

    __AGENCY_ANNOTATION
    value_type& operator[](index_type idx)
    {
      return all_[idx];
    }

    __AGENCY_ANNOTATION
    shape_type shape() const
    {
      return all_.shape();
    }

    __AGENCY_ANNOTATION
    std::size_t size() const
    {
      return all_.size();
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return all_.data();
    }

    __AGENCY_ANNOTATION
    const_pointer data() const
    {
      return all_.data();
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

    __agency_exec_check_disable__
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

        alloc_.deallocate(data(), size());

        all_ = all_t{};
      }
    }

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const basic_ndarray& lhs, const Range& rhs)
    {
      if(lhs.size() != rhs.size()) return false;

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

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const Range& lhs, const basic_ndarray& rhs)
    {
      return rhs == lhs;
    }

    // this operator== avoids ambiguities introduced by the template friends above
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bool operator==(const basic_ndarray& rhs) const
    {
      if(size() != rhs.size()) return false;

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
    __agency_exec_check_disable__
    template<class... Args>
    __AGENCY_ANNOTATION
    static pointer allocate_and_construct_elements(allocator_type& alloc, size_t size, const Args&... args)
    {
      pointer result = alloc.allocate(size);

      agency::detail::allocator_traits<allocator_type>::construct_n(alloc, result, size, agency::detail::constant_iterator<Args>(args,0)...);

      return result;
    }

    allocator_type alloc_;

    all_t all_;
};


template<class T, size_t rank, class Alloc = std::allocator<T>>
using ndarray = basic_ndarray<T, point<std::size_t,rank>, Alloc>;


} // end experimental
} // end agency

