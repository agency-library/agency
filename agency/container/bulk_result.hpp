#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/executor_index.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_allocator.hpp>
#include <agency/memory/detail/storage.hpp>
#include <agency/memory/allocator/allocator.hpp>
#include <agency/coordinate/detail/colexicographic_rank.hpp>

namespace agency
{


template<class T, class Shape, class Allocator = allocator<T>>
class bulk_result : private detail::storage<T, Allocator, Shape>
{
  private:
    using super_t = detail::storage<T, Allocator, Shape>;

  public:
    using value_type = T;
    using shape_type = Shape;
    using index_type = Shape;
    using allocator_type = Allocator;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    // XXX this should be eliminated
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bulk_result()
      : bulk_result(shape_type{})
    {}

    // XXX this should be made private
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit bulk_result(const shape_type& shape, const allocator_type& alloc = allocator_type())
      : super_t(shape, alloc)
    {
      // XXX we shouldn't actually construct the elements
      //     bulk_invoke et al. should placement new them as they get assigned through operator[]
      construct_elements();
    }

    // XXX this should be eliminated
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bulk_result(const shape_type& shape, const T& val, const allocator_type& alloc = allocator_type())
      : super_t(shape, alloc)
    {
      construct_elements(val);
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bulk_result(bulk_result&& other)
      : super_t(std::move(other))
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bulk_result(const bulk_result& other)
      : bulk_result(other.shape(), other.allocator())
    {
      value_type* ptr = super_t::data();
      value_type* end = ptr + super_t::size();

      const value_type* other_ptr = other.data();

      for(; ptr != end; ++ptr, ++other_ptr)
      {
        agency::detail::allocator_traits<allocator_type>::construct(super_t::allocator(), ptr, *other_ptr);
      }
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    ~bulk_result()
    {
      value_type* ptr = super_t::data();
      value_type* end = ptr + super_t::size();

      for(; ptr != end; ++ptr)
      {
        agency::detail::allocator_traits<allocator_type>::destroy(super_t::allocator(), ptr);
      }
    }

    __AGENCY_ANNOTATION
    bulk_result& operator=(const bulk_result& other)
    {
      // XXX this is not a very efficient implementation
      bulk_result tmp = other;
      swap(tmp);
      return *this;
    }

    __AGENCY_ANNOTATION
    bulk_result& operator=(bulk_result&& other)
    {
      swap(other);
      return *this;
    }

    __AGENCY_ANNOTATION
    value_type& operator[](index_type idx)
    {
      auto rank = agency::detail::colexicographic_rank(idx, shape());
      return super_t::data()[rank];
    }

    __AGENCY_ANNOTATION
    shape_type shape() const
    {
      return super_t::shape();
    }

    __AGENCY_ANNOTATION
    std::size_t size() const
    {
      return super_t::size();
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return super_t::data();
    }

    __AGENCY_ANNOTATION
    const_pointer data() const
    {
      return super_t::data();
    }

    __AGENCY_ANNOTATION
    iterator begin()
    {
      return data();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return data();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return begin() + size();
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return begin() + size();
    }

    __AGENCY_ANNOTATION
    void swap(bulk_result& other)
    {
      super_t::swap(other);
    }

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    bool operator==(const Range& rhs) const
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
    void construct_elements(const Args&... args)
    {
      value_type* ptr = super_t::data();
      value_type* end = ptr + super_t::size();

      for(; ptr != end; ++ptr)
      {
        agency::detail::allocator_traits<allocator_type>::construct(super_t::allocator(), ptr, args...);
      }
    }
};


} // end agency

