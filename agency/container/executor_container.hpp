#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/executor_index.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_allocator.hpp>
#include <agency/memory/detail/storage.hpp>
#include <agency/detail/index_lexicographical_rank.hpp>

namespace agency
{


// XXX eliminate this alias in favor of a type e.g. bulk_result
template<class Executor, class T>
class executor_container : private detail::storage<T, executor_allocator_t<Executor,T>, executor_shape_t<Executor>>
{
  private:
    using super_t = detail::storage<T, executor_allocator_t<Executor,T>, executor_shape_t<Executor>>;

  public:
    using value_type = T;
    using shape_type = executor_shape_t<Executor>;
    using index_type = executor_index_t<Executor>;
    using allocator_type = executor_allocator_t<Executor,T>;
    using iterator = T*;
    using const_iterator = const T*;

    // XXX this should be eliminated
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container()
      : executor_container(shape_type{})
    {}

    // XXX this should be made private
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit executor_container(const shape_type& shape, const allocator_type& alloc = allocator_type())
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
    executor_container(const shape_type& shape, const T& val, const allocator_type& alloc = allocator_type())
      : super_t(shape, alloc)
    {
      construct_elements(val);
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container(executor_container&& other)
      : super_t(std::move(other))
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container(const executor_container& other)
      : executor_container(other.shape(), other.allocator())
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
    ~executor_container()
    {
      value_type* ptr = super_t::data();
      value_type* end = ptr + super_t::size();

      for(; ptr != end; ++ptr)
      {
        agency::detail::allocator_traits<allocator_type>::destroy(super_t::allocator(), ptr);
      }
    }

    __AGENCY_ANNOTATION
    executor_container& operator=(const executor_container& other)
    {
      // XXX this is not a very efficient implementation
      executor_container tmp = other;
      swap(tmp);
      return *this;
    }

    __AGENCY_ANNOTATION
    executor_container& operator=(executor_container&& other)
    {
      swap(other);
      return *this;
    }

    __AGENCY_ANNOTATION
    value_type& operator[](index_type idx)
    {
      auto rank = agency::detail::index_lexicographical_rank(idx, shape());
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
    iterator begin()
    {
      return super_t::data();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return super_t::data();
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
    void swap(executor_container& other)
    {
      super_t::swap(other);
    }

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const executor_container& lhs, const Range& rhs)
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
    friend bool operator==(const Range& lhs, const executor_container& rhs)
    {
      return rhs == lhs;
    }

    // this operator== avoids ambiguities introduced by the template friends above
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bool operator==(const executor_container& rhs) const
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

