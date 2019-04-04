#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/default_shape.hpp>
#include <agency/experimental/ndarray/ndarray_ref.hpp>
#include <agency/memory/allocator/allocator.hpp>
#include <agency/detail/iterator/constant_iterator.hpp>
#include <agency/execution/execution_policy/detail/simple_sequenced_policy.hpp>
#include <agency/detail/algorithm/construct_n.hpp>
#include <agency/detail/algorithm/bulk_construct.hpp>
#include <agency/detail/algorithm/destroy.hpp>
#include <agency/detail/algorithm/bulk_destroy.hpp>
#include <agency/detail/algorithm/equal.hpp>
#include <agency/experimental/ndarray/constant_ndarray.hpp>
#include <agency/experimental/ranges/all.hpp>
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

  public:
    using value_type = T;
    using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<value_type>;
    using size_type = typename std::allocator_traits<allocator_type>::size_type;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

    using shape_type = typename storage_type::shape_type;
    using index_type = Index;

    using iterator = pointer;
    using const_iterator = const_pointer;

    using reference = typename std::iterator_traits<iterator>::reference;
    using const_reference = typename std::iterator_traits<const_iterator>::reference;

    // note that basic_ndarray's constructors have __agency_exec_check_disable__
    // because Alloc's constructors may not have __AGENCY_ANNOTATION

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit basic_ndarray(const allocator_type& alloc)
      : storage_(shape_type{}, alloc)
    {}

    // XXX we should reformulate all the following constructors such that they lower onto this
    //     general purpose constructor
    __agency_exec_check_disable__
    template<class... Args, __AGENCY_REQUIRES(std::is_constructible<T, const Args&...>::value)>
    __AGENCY_ANNOTATION
    basic_ndarray(const shape_type& shape, const allocator_type& alloc, const Args&... constructor_args)
      : storage_(shape, alloc)
    {
      construct_elements_from_arrays(constant_ndarray<Args,Shape>(shape, constructor_args)...);
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray() : basic_ndarray(allocator_type()) {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit basic_ndarray(const shape_type& shape, const allocator_type& alloc = allocator_type())
      : storage_(shape, alloc)
    {
      construct_elements_from_arrays();
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray(const shape_type& shape, const T& val, const allocator_type& alloc = allocator_type())
      : basic_ndarray(constant_ndarray<T,Shape>(shape, val), alloc)
    {}

    template<class Array,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 storage_type, decltype(std::declval<Array&&>().shape()), allocator_type
               >::value
             )>
    __AGENCY_ANNOTATION
    explicit basic_ndarray(Array&& array, const allocator_type& alloc)
      : storage_(array.shape(), alloc)
    {
      construct_elements_from_arrays(std::forward<Array>(array));
    }

    __agency_exec_check_disable__
    template<class Array,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 storage_type, decltype(std::declval<Array&>().shape()), allocator_type
               >::value and
               std::is_default_constructible<
                 allocator_type
               >::value
             )>
    __AGENCY_ANNOTATION
    explicit basic_ndarray(Array&& array)
      : basic_ndarray(std::forward<Array>(array), allocator_type())
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
      : basic_ndarray(agency::detail::simple_sequenced_policy<>(), first, shape, alloc)
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
      construct_elements_from_arrays(std::forward<ExecutionPolicy>(policy), other.all());
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray(const basic_ndarray& other)
      : storage_(other.shape(), other.get_allocator())
    {
      construct_elements_from_arrays(other.all());
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_ndarray(basic_ndarray&& other)
      : storage_{std::move(other.storage_)}
    {}

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
    reference operator[](index_type idx)
    {
      return all()[idx];
    }

    __AGENCY_ANNOTATION
    const_reference operator[](index_type idx) const
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
    bool empty() const
    {
      return size() == 0;
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
    basic_ndarray_ref<const_pointer,shape_type,index_type> all() const
    {
      return basic_ndarray_ref<const_pointer,shape_type,index_type>(data(), shape());
    }

    __AGENCY_ANNOTATION
    basic_ndarray_ref<pointer,shape_type,index_type> all()
    {
      return basic_ndarray_ref<pointer,shape_type,index_type>(data(), shape());
    }

    __AGENCY_ANNOTATION
    iterator begin()
    {
      return all().data();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return all().end();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return all().begin();
    }

    __AGENCY_ANNOTATION
    const_iterator cbegin() const
    {
      return begin();
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return all().end();
    }

    __AGENCY_ANNOTATION
    const_iterator cend() const
    {
      return end();
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    void clear()
    {
      agency::detail::bulk_destroy(storage_.allocator(), all());

      // reset the storage to empty
      storage_ = storage_type(std::move(storage_.allocator()));
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
      agency::detail::construct_n(std::forward<ExecutionPolicy>(policy), storage_.allocator(), begin(), size(), iters...);
    }

    template<class... Iterators>
    __AGENCY_ANNOTATION
    void construct_elements(Iterators... iters)
    {
      agency::detail::simple_sequenced_policy<> seq;
      construct_elements(seq, iters...);
    }


    template<class ExecutionPolicy, class... Arrays,
             __AGENCY_REQUIRES(
               is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
             )>
    __AGENCY_ANNOTATION
    void construct_elements_from_arrays(ExecutionPolicy&& policy, Arrays&&... arrays)
    {
      agency::detail::bulk_construct(storage_.allocator(), std::forward<ExecutionPolicy>(policy), all(), experimental::all(std::forward<Arrays>(arrays))...);
    }

    template<class... Arrays>
    __AGENCY_ANNOTATION
    void construct_elements_from_arrays(Arrays&&... arrays)
    {
      agency::detail::bulk_construct(storage_.allocator(), all(), experimental::all(std::forward<Arrays>(arrays))...);
    }

    storage_type storage_;
};


template<class T, size_t rank, class Alloc = agency::allocator<T>>
using ndarray = basic_ndarray<T, agency::detail::default_shape_t<rank>, Alloc>;


} // end experimental
} // end agency

