#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/executor_index.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_allocator.hpp>
#include <agency/experimental/ndarray.hpp>

namespace agency
{


// XXX eliminate this alias in favor of a type e.g. bulk_result
template<class Executor, class T>
class executor_container : private experimental::basic_ndarray<T, executor_shape_t<Executor>, executor_allocator_t<Executor,T>, executor_index_t<Executor>>
{
  private:
    using super_t = experimental::basic_ndarray<T, executor_shape_t<Executor>, executor_allocator_t<Executor,T>, executor_index_t<Executor>>;

  public:
    using value_type = T;
    using shape_type = executor_shape_t<Executor>;
    using index_type = executor_index_t<Executor>;
    using allocator_type = executor_allocator_t<Executor,T>;
    using iterator = typename super_t::pointer;
    using const_iterator = typename super_t::const_pointer;

    // XXX this should be eliminated
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container() = default;

    // XXX this should be eliminated
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit executor_container(const shape_type& shape, const allocator_type& alloc = allocator_type())
      : super_t(shape, alloc)
    {}

    // XXX this should be eliminated
    //     it should not really be possible to create these things except via bulk_invoke et al.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container(const shape_type& shape, const T& val, const allocator_type& alloc = allocator_type())
      : super_t(shape, val, alloc)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container(executor_container&& other)
      : super_t(std::move(other))
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container(const executor_container& other)
      : super_t(other)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    ~executor_container() = default;

    __AGENCY_ANNOTATION
    executor_container& operator=(const executor_container& other)
    {
      super_t::operator=(other);
      return *this;
    }

    __AGENCY_ANNOTATION
    executor_container& operator=(executor_container&& other)
    {
      super_t::operator=(std::move(other));
      return *this;
    }

    __AGENCY_ANNOTATION
    value_type& operator[](index_type idx)
    {
      return super_t::operator[](idx);
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
      return super_t::begin();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return super_t::begin();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return super_t::end();
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return super_t::end();
    }

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const executor_container& lhs, const Range& rhs)
    {
      return static_cast<const super_t&>(lhs) == rhs;
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
      return super_t::operator==(rhs);
    }
};


} // end agency

