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
    using shape_type = typename super_t::shape_type;
    using allocator_type = typename super_t::allocator_type;

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
    executor_container(executor_container&& other) = default;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_container(const executor_container& other) = default;

    __AGENCY_ANNOTATION
    ~executor_container() = default;

    __AGENCY_ANNOTATION
    executor_container& operator=(const executor_container&) = default;

    __AGENCY_ANNOTATION
    executor_container& operator=(executor_container&&) = default;

    using super_t::operator[];
    using super_t::shape;
    using super_t::size;
    using super_t::begin;
    using super_t::end;

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

