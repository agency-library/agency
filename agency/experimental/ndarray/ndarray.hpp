#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/default_shape.hpp>
#include <agency/experimental/ndarray/ndarray_ref.hpp>
#include <agency/memory/allocator/allocator.hpp>
#include <agency/detail/iterator/constant_iterator.hpp>
#include <agency/execution/execution_policy/detail/simple_sequenced_policy.hpp>
#include <agency/detail/algorithm/construct_n.hpp>
#include <agency/detail/algorithm/destroy.hpp>
#include <agency/detail/algorithm/equal.hpp>
#include <utility>
#include <memory>
#include <iterator>


namespace agency
{
namespace experimental
{


template<class T, class Shape = size_t, class Alloc = agency::allocator<T>, class Index = Shape>
class basic_ndarray
{
  private:
    using storage_type = agency::detail::storage<T,Alloc,Shape>;
    using all_t = basic_ndarray_ref<T,Shape,Index>;

  public:
    using value_type = T;

    using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<value_type>;

    using pointer = typename std::allocator_traits<allocator_type>::pointer;

    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

    using shape_type = typename storage_type::shape_type;

    using index_type = typename all_t::index_type;

    // note that basic_ndarray's constructors have __agency_exec_check_disable__
    // because Alloc's constructors may not have __AGENCY_ANNOTATION

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray() : basic_ndarray(shape_type{}) {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit basic_ndarray(const shape_type& shape, const allocator_type& alloc = allocator_type())
      : storage_(shape, alloc)
    {
      construct_elements();
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit basic_ndarray(const shape_type& shape, const T& val, const allocator_type& alloc = allocator_type())
      : basic_ndarray(agency::detail::constant_iterator<T>(val,0),
                      agency::detail::constant_iterator<T>(val, agency::detail::index_space_size(shape)),
                      alloc)
    {}

    template<class ExecutionPolicy,
             class Iterator,
             __AGENCY_REQUIRES(
               is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
             ),
             // XXX this requirement should really be something like is_input_iterator<InputIterator>
             __AGENCY_REQUIRES(
               std::is_convertible<typename std::iterator_traits<Iterator>::value_type, value_type>::value
             )>
    basic_ndarray(ExecutionPolicy&& policy, Iterator first, shape_type shape, const allocator_type& alloc = allocator_type())
      : storage_(shape, alloc)
    {
      construct_elements(std::forward<ExecutionPolicy>(policy), first);
    }

    template<class Iterator,
             // XXX this requirement should really be something like is_input_iterator<InputIterator>
             __AGENCY_REQUIRES(
               std::is_convertible<typename std::iterator_traits<Iterator>::value_type, value_type>::value
             )>
    basic_ndarray(Iterator first, shape_type shape, const allocator_type& alloc = allocator_type())
      : basic_ndarray(agency::detail::simple_sequenced_policy(), first, shape, alloc)
    {}

    __agency_exec_check_disable__
    template<class Iterator,
             // XXX this requirement should really be something like is_input_iterator<InputIterator>
             __AGENCY_REQUIRES(
               std::is_convertible<typename std::iterator_traits<Iterator>::value_type, value_type>::value
             )>
    __AGENCY_ANNOTATION
    basic_ndarray(Iterator first, Iterator last, const allocator_type& alloc = allocator_type())
      : basic_ndarray(first, agency::detail::shape_cast<shape_type>(last - first), alloc)
    {}

    __agency_exec_check_disable__
    template<class ExecutionPolicy,
             __AGENCY_REQUIRES(
               is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
             )>
    __AGENCY_ANNOTATION
    basic_ndarray(ExecutionPolicy&& policy, const basic_ndarray& other)
      : storage_(other.shape(), other.get_allocator())
    {
      construct_elements(std::forward<ExecutionPolicy>(policy), other.begin());
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray(const basic_ndarray& other)
      : basic_ndarray(agency::detail::simple_sequenced_policy(), other)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray(basic_ndarray&& other)
      : storage_{}
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
      storage_.swap(other.storage_);
    }


    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    allocator_type get_allocator() const
    {
      return storage_.allocator();
    }

    __AGENCY_ANNOTATION
    value_type& operator[](index_type idx)
    {
      return all()[idx];
    }

    __AGENCY_ANNOTATION
    shape_type shape() const
    {
      return storage_.shape();
    }

    __AGENCY_ANNOTATION
    std::size_t size() const
    {
      return storage_.size();
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return storage_.data();
    }

    __AGENCY_ANNOTATION
    const_pointer data() const
    {
      return storage_.data();
    }

    __AGENCY_ANNOTATION
    basic_ndarray_ref<const T,Shape,Index> all() const
    {
      return basic_ndarray_ref<const T,Shape,Index>(data(), shape());
    }

    __AGENCY_ANNOTATION
    basic_ndarray_ref<T,Shape,Index> all()
    {
      return basic_ndarray_ref<T,Shape,Index>(data(), shape());
    }

    __AGENCY_ANNOTATION
    pointer begin()
    {
      return all().data();
    }

    __AGENCY_ANNOTATION
    pointer end()
    {
      return all().end();
    }

    __AGENCY_ANNOTATION
    const_pointer begin() const
    {
      return all().begin();
    }

    __AGENCY_ANNOTATION
    const_pointer cbegin() const
    {
      return begin();
    }

    __AGENCY_ANNOTATION
    const_pointer end() const
    {
      return all().end();
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
      agency::detail::destroy(storage_.allocator(), begin(), end());
      storage_ = storage_type{};
    }

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const basic_ndarray& lhs, const Range& rhs)
    {
      return lhs.size() == rhs.size() && agency::detail::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    __agency_exec_check_disable__
    template<class Range>
    __AGENCY_ANNOTATION
    friend bool operator==(const Range& lhs, const basic_ndarray& rhs)
    {
      return rhs == lhs;
    }

    // this operator== avoids ambiguities introduced by the template friends above
    __AGENCY_ANNOTATION
    bool operator==(const basic_ndarray& rhs) const
    {
      return size() == rhs.size() && agency::detail::equal(begin(), end(), rhs.begin());
    }

  private:
    template<class ExecutionPolicy, class... Iterators,
             __AGENCY_REQUIRES(
               is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
             )>
    __AGENCY_ANNOTATION
    void construct_elements(ExecutionPolicy&& policy, Iterators... iters)
    {
      agency::detail::construct_n(std::forward<ExecutionPolicy>(policy), begin(), size(), iters...);
    }

    template<class... Iterators>
    __AGENCY_ANNOTATION
    void construct_elements(Iterators... iters)
    {
      agency::detail::simple_sequenced_policy seq;
      construct_elements(seq, iters...);
    }

    storage_type storage_;
};


template<class T, size_t rank, class Alloc = agency::allocator<T>>
using ndarray = basic_ndarray<T, agency::detail::default_shape_t<rank>, Alloc>;


} // end experimental
} // end agency

