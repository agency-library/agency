#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/detail/utility.hpp>
#include <agency/detail/iterator.hpp>
#include <agency/detail/algorithm.hpp>
#include <agency/experimental/memory/allocator.hpp>
#include <memory>
#include <initializer_list>

namespace agency
{
namespace experimental
{
namespace detail
{


__AGENCY_ANNOTATION
inline void throw_length_error(const char* what_arg)
{
#ifdef __CUDA_ARCH__
  printf("length_error: %s\n", what_arg);
  assert(0);
#else
  throw std::length_error(what_arg);
#endif
}


__AGENCY_ANNOTATION
inline void throw_out_of_range(const char* what_arg)
{
#ifdef __CUDA_ARCH__
  printf("out_of_range: %s\n", what_arg);
  assert(0);
#else
  throw std::out_of_range(what_arg);
#endif
}


// XXX move this underneath detail/container/storage.hpp
template<class T, class Allocator>
class storage
{
  public:
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    storage(size_t count, const Allocator& allocator)
      : data_(nullptr),
        size_(count),
        allocator_(allocator)
    {
      if(count > 0)
      {
        data_ = agency::detail::allocator_traits<Allocator>::allocate(allocator_, count);
        if(data_ == nullptr)
        {
          detail::throw_bad_alloc();
        }
      }
    }

    __AGENCY_ANNOTATION
    storage(size_t count, Allocator&& allocator)
      : data_(nullptr),
        size_(count),
        allocator_(std::move(allocator))
    {
      if(count > 0)
      {
        data_ = allocator_.allocate(count);
        if(data_ == nullptr)
        {
          detail::throw_bad_alloc();
        }
      }
    }

    __AGENCY_ANNOTATION
    storage(storage&& other)
      : data_(other.data_),
        size_(other.size_),
        allocator_(std::move(other.allocator_))
    {
      // leave the other storage in a valid state
      other.data_ = nullptr;
      other.size_ = 0;
    }

    __AGENCY_ANNOTATION
    storage(const Allocator& allocator)
      : storage(0, allocator)
    {}

    __AGENCY_ANNOTATION
    storage()
      : storage(Allocator())
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    ~storage()
    {
      agency::detail::allocator_traits<Allocator>::deallocate(allocator_, data(), size());
    }

  private:
    __AGENCY_ANNOTATION
    void move_assign_allocator(std::true_type, Allocator& other_allocator)
    {
      // propagate the allocator
      allocator_ = std::move(other_allocator);
    }

    __AGENCY_ANNOTATION
    void move_assign_allocator(std::false_type, Allocator&)
    {
      // do nothing
    }

  public:
    __AGENCY_ANNOTATION
    storage& operator=(storage&& other)
    {
      agency::detail::adl_swap(data_, other.data_);
      agency::detail::adl_swap(size_, other.size_);

      move_assign_allocator(typename std::allocator_traits<Allocator>::propagate_on_container_move_assignment(), other.allocator());
    }

    __AGENCY_ANNOTATION
    T* data()
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    const T* data() const
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    size_t size() const
    {
      return size_;
    }

    __AGENCY_ANNOTATION
    const Allocator& allocator() const
    {
      return allocator_;
    }

    __AGENCY_ANNOTATION
    Allocator& allocator()
    {
      return allocator_;
    }

    __AGENCY_ANNOTATION
    void swap(storage& other)
    {
      agency::detail::adl_swap(data_, other.data_);
      agency::detail::adl_swap(size_, other.size_);
      agency::detail::adl_swap(allocator_, other.allocator_);
    }

  private:
    T* data_;
    size_t size_;
    Allocator allocator_;
};


} // end detail


template<class T, class Allocator = allocator<T>>
class vector
{
  private:
    using storage_type = detail::storage<T,Allocator>;

  public:
    using allocator_type  = Allocator;
    using value_type      = typename agency::detail::allocator_traits<allocator_type>::value_type;
    using size_type       = typename agency::detail::allocator_traits<allocator_type>::size_type;
    using difference_type = typename agency::detail::allocator_traits<allocator_type>::difference_type;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = typename agency::detail::allocator_traits<allocator_type>::pointer;
    using const_pointer   = typename agency::detail::allocator_traits<allocator_type>::const_pointer;

    using iterator = pointer;
    using const_iterator = const_pointer;

    using reverse_iterator = void;
    using const_reverse_iterator = void;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    vector() : vector(Allocator()) {}

    __AGENCY_ANNOTATION
    explicit vector(const Allocator& alloc)
      : storage_(alloc), end_(begin())
    {}

    __AGENCY_ANNOTATION
    vector(size_type count, const T& value, const Allocator& alloc = Allocator())
      : vector(agency::sequenced_execution_policy(), count, value, alloc)
    {}

    template<class ExecutionPolicy, __AGENCY_REQUIRES(is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value)>
    __AGENCY_ANNOTATION
    vector(ExecutionPolicy&& policy, size_type count, const T& value, const Allocator& alloc = Allocator())
      : vector(std::forward<ExecutionPolicy>(policy), agency::detail::constant_iterator<T>(value,0), agency::detail::constant_iterator<T>(value,count), alloc)
    {}

    __AGENCY_ANNOTATION
    explicit vector(size_type count, const Allocator& alloc = Allocator())
      : vector(agency::sequenced_execution_policy(), count, alloc)
    {}

    template<class ExecutionPolicy, __AGENCY_REQUIRES(is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value)>
    __AGENCY_ANNOTATION
    vector(ExecutionPolicy&& policy, size_type count, const Allocator& alloc = Allocator())
      : vector(std::forward<ExecutionPolicy>(policy), count, T(), alloc)
    {}

    template<class InputIterator,
             __AGENCY_REQUIRES(
               std::is_convertible<
                 typename std::iterator_traits<InputIterator>::iterator_category,
                 std::input_iterator_tag
               >::value
             )>
    __AGENCY_ANNOTATION
    vector(InputIterator first, InputIterator last, const Allocator& alloc = Allocator())
      : vector(agency::sequenced_execution_policy(), first, last, alloc)
    {}

    // this is the most fundamental constructor
    template<class ExecutionPolicy,
             class InputIterator,
             __AGENCY_REQUIRES(
               std::is_convertible<
                 typename std::iterator_traits<InputIterator>::iterator_category,
                 std::input_iterator_tag
               >::value
             )>
    __AGENCY_ANNOTATION
    vector(ExecutionPolicy&& policy, InputIterator first, InputIterator last, const Allocator& alloc = Allocator())
      : storage_(alloc), // initialize the storage to empty
        end_(begin())    // initialize end_ to begin()
    {
      insert(std::forward<ExecutionPolicy>(policy), end(), first, last);
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    vector(const vector& other)
      : vector(agency::sequenced_execution_policy(), other, other.get_allocator())
    {}

    __agency_exec_check_disable__
    template<class ExecutionPolicy>
    __AGENCY_ANNOTATION
    vector(ExecutionPolicy&& policy, const vector& other)
      : vector(std::forward<ExecutionPolicy>(policy), other, other.get_allocator())
    {}

    __AGENCY_ANNOTATION
    vector(const vector& other, const Allocator& alloc)
      : vector(agency::sequenced_execution_policy(), other.begin(), other.end(), alloc)
    {}

    template<class ExecutionPolicy>
    __AGENCY_ANNOTATION
    vector(ExecutionPolicy&& policy, const vector& other, const Allocator& alloc)
      : vector(std::forward<ExecutionPolicy>(policy), other.begin(), other.end(), alloc)
    {}

    __AGENCY_ANNOTATION
    vector(vector&& other)
      : storage_(std::move(other.storage_)),
        end_(other.end_)
    {
      // leave the other vector in a valid state
      other.end_ = other.begin();
    }

    __AGENCY_ANNOTATION
    vector(vector&& other, const Allocator& alloc)
      : storage_(std::move(other.storage_), alloc),
        end_(other.end_)
    {}

    __AGENCY_ANNOTATION
    vector(std::initializer_list<T> init, const Allocator& alloc = Allocator())
      : vector(init.begin(), init.end(), alloc)
    {}

    __AGENCY_ANNOTATION
    ~vector()
    {
      clear();
    }

    __AGENCY_ANNOTATION
    vector& operator=(const vector& other)
    {
      assign(other.begin(), other.end());
      return *this;
    }

    __AGENCY_ANNOTATION
    vector& operator=(vector&& other)
    {
      storage_ = std::move(other.storage_);
      agency::detail::adl_swap(end_, other.end_);
      return *this;
    }

    __AGENCY_ANNOTATION
    vector& operator=(std::initializer_list<T> ilist)
    {
      assign(ilist.begin(), ilist.end());
      return *this;
    }

    __AGENCY_ANNOTATION
    void assign(size_type count, const T& value)
    {
      assign(agency::sequenced_execution_policy(), count, value);
    }

    template<class ExecutionPolicy, __AGENCY_REQUIRES(is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value)>
    __AGENCY_ANNOTATION
    void assign(ExecutionPolicy&& policy, size_type count, const T& value)
    {
      assign(std::forward<ExecutionPolicy>(policy), agency::detail::constant_iterator<T>(value,0), agency::detail::constant_iterator<T>(value,count));
    }

  private:
    template<class ExecutionPolicy, class ForwardIterator>
    __AGENCY_ANNOTATION
    void assign(std::forward_iterator_tag, ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last)
    {
      size_type n = agency::detail::distance(first, last);

      if(n > capacity())
      {
        // n is too large for capacity, swap with a new vector
        vector new_vector(policy, first, last);
        swap(new_vector);
      }
      else if(size() >= n)
      {
        // we can already accomodate the new range
        iterator old_end = end();
        end_ = agency::detail::copy(policy, first, last, begin());

        // destroy the old elements
        agency::detail::destroy(policy, storage_.allocator(), end(), old_end);
      }
      else
      {
        // range fits inside allocated storage

        // copy to already existing elements
        auto mid_and_end = agency::detail::copy_n(policy, first, size(), begin());

        // construct new elements at the end
        end_ = agency::detail::uninitialized_copy_n(policy, storage_.allocator(), agency::detail::get<0>(mid_and_end), n - size(), end());
      }
    }

    template<class ExecutionPolicy, class InputIterator>
    __AGENCY_ANNOTATION
    void assign(std::input_iterator_tag, ExecutionPolicy&& policy, InputIterator first, InputIterator last)
    {
      iterator current = begin();

      // assign to elements which already exist
      for(; first != last && current != end(); ++current, ++first)
      {
        *current = *first;
      }

      // either only the input was exhausted or both
      // the the input and vector elements were exhaused
      if(first == last)
      {
        // if we exhausted the input, erase leftover elements
        erase(policy, current, end());
      }
      else
      {
        // insert the rest of the input at the end of the vector
        insert(policy, end(), first, last);
      }
    }

  public:
    template<class InputIterator>
    __AGENCY_ANNOTATION
    void assign(InputIterator first, InputIterator last)
    {
      assign(agency::sequenced_execution_policy(), first, last);
    }

    template<class ExecutionPolicy, class InputIterator,
             __AGENCY_REQUIRES(is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value)>
    __AGENCY_ANNOTATION
    void assign(ExecutionPolicy&& policy, InputIterator first, InputIterator last)
    {
      assign(typename std::iterator_traits<InputIterator>::iterator_category(), std::forward<ExecutionPolicy>(policy), first, last);
    }

    __AGENCY_ANNOTATION
    void assign(std::initializer_list<T> ilist)
    {
      assign(ilist.begin(), ilist.end());
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    allocator_type get_allocator() const
    {
      return storage_.allocator();
    }

    // element access
    
    __AGENCY_ANNOTATION
    reference at(size_type pos)
    {
      if(pos >= size())
      {
        detail::throw_out_of_range("pos >= size() in vector::at()");
      }

      return operator[](pos);
    }

    __AGENCY_ANNOTATION
    const_reference at(size_type pos) const
    {
      if(pos >= size())
      {
        detail::throw_out_of_range("pos >= size() in vector::at()");
      }

      return operator[](pos);
    }

    __AGENCY_ANNOTATION
    reference operator[](size_type pos)
    {
      return begin()[pos];
    }

    __AGENCY_ANNOTATION
    const_reference operator[](size_type pos) const
    {
      return begin()[pos];
    }

    __AGENCY_ANNOTATION
    reference front()
    {
      return *begin();
    }

    __AGENCY_ANNOTATION
    const_reference front() const
    {
      return *begin();
    }

    __AGENCY_ANNOTATION
    reference back()
    {
      return *(end()-1);
    }

    __AGENCY_ANNOTATION
    const_reference back() const
    {
      return *(end()-1);
    }

    __AGENCY_ANNOTATION
    T* data()
    {
      return storage_.data();
    }

    __AGENCY_ANNOTATION
    const T* data() const
    {
      return storage_.data();
    }
    
    // iterators

    __AGENCY_ANNOTATION
    iterator begin()
    {
      return storage_.data();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return cbegin();
    }

    __AGENCY_ANNOTATION
    const_iterator cbegin() const
    {
      return storage_.data();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return end_;
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return cend();
    }

    __AGENCY_ANNOTATION
    const_iterator cend() const
    {
      return end_;
    }

    // TODO
    __AGENCY_ANNOTATION
    reverse_iterator rbegin();

    // TODO
    __AGENCY_ANNOTATION
    const_reverse_iterator rbegin() const;

    // TODO
    __AGENCY_ANNOTATION
    const_reverse_iterator crbegin() const;

    // TODO
    __AGENCY_ANNOTATION
    reverse_iterator rend();

    // TODO
    __AGENCY_ANNOTATION
    const_reverse_iterator rend() const;

    // TODO
    __AGENCY_ANNOTATION
    const_reverse_iterator crend() const;

    // capacity

    __AGENCY_ANNOTATION
    bool empty() const
    {
      return cbegin() == cend();
    }

    __AGENCY_ANNOTATION
    size_type size() const
    {
      return end() - begin();
    }

    __AGENCY_ANNOTATION
    size_type max_size() const
    {
      return agency::detail::allocator_traits<allocator_type>::max_size(storage_.allocator());
    }

    // XXX this needs an ExecutionPolicy overload
    __AGENCY_ANNOTATION
    void reserve(size_type new_capacity)
    {
      if(new_capacity > capacity())
      {
        if(new_capacity > max_size())
        {
          detail::throw_length_error("reserve(): new capacity exceeds max_size().");
        }

        // create a new storage object
        storage_type new_storage(new_capacity, storage_.allocator());

        // copy our elements into the new storage
        end_ = agency::detail::uninitialized_copy(new_storage.allocator(), begin(), end(), new_storage.data());

        // swap out our storage
        storage_.swap(new_storage);
      }
    }

    __AGENCY_ANNOTATION
    size_type capacity() const
    {
      return storage_.size();
    }

    // XXX this needs an ExecutionPolicy overload
    __AGENCY_ANNOTATION
    void shrink_to_fit()
    {
      vector(*this).swap(*this);
    }

    // modifiers
    
    // XXX this needs an ExecutionPolicy overload
    __AGENCY_ANNOTATION
    void clear()
    {
      agency::detail::destroy(storage_.allocator(), begin(), end());
      end_ = begin();
    }

    // single element insert

    __AGENCY_ANNOTATION
    iterator insert(const_iterator position, const T& value)
    {
      return emplace(position, value);
    }

    __AGENCY_ANNOTATION
    iterator insert(const_iterator position, T&& value)
    {
      return emplace(position, std::move(value));
    }

    // fill insert

    template<class ExecutionPolicy>
    __AGENCY_ANNOTATION
    iterator insert(ExecutionPolicy&& policy, const_iterator position, size_type count, const T& value)
    {
      return insert(std::forward<ExecutionPolicy>(policy), position, agency::detail::constant_iterator<T>(value,0), agency::detail::constant_iterator<T>(value,count));
    }

    __AGENCY_ANNOTATION
    iterator insert(const_iterator position, size_type count, const T& value)
    {
      agency::sequenced_execution_policy seq;
      return insert(seq, position, count, value);
    }

    template<class ExecutionPolicy,
             class ForwardIterator,
             __AGENCY_REQUIRES(
               std::is_convertible<
                 typename std::iterator_traits<ForwardIterator>::iterator_category,
                 std::forward_iterator_tag
               >::value
             )
            >
    __AGENCY_ANNOTATION
    iterator insert(ExecutionPolicy&& policy, const_iterator position, ForwardIterator first, ForwardIterator last)
    {
      return emplace_n(std::forward<ExecutionPolicy>(policy), position, agency::detail::distance(first, last), first);
    }

    // range insert

    template<class ForwardIterator,
             __AGENCY_REQUIRES(
               std::is_convertible<
                 typename std::iterator_traits<ForwardIterator>::iterator_category,
                 std::forward_iterator_tag
               >::value
             )
            >
    __AGENCY_ANNOTATION
    iterator insert(const_iterator position, ForwardIterator first, ForwardIterator last)
    {
      agency::sequenced_execution_policy seq;
      return insert(seq, position, first, last);
    }

    template<class InputIterator,
             __AGENCY_REQUIRES(
               !std::is_convertible<
                 typename std::iterator_traits<InputIterator>::iterator_category,
                 std::forward_iterator_tag
               >::value
             )
            >
    __AGENCY_ANNOTATION
    iterator insert(const_iterator position, InputIterator first, InputIterator last)
    {
      for(; first != last; ++first)
      {
        position = insert(position, *first);
      }

      return position;
    }

    __AGENCY_ANNOTATION
    iterator insert(const_iterator pos, std::initializer_list<T> ilist)
    {
      return insert(pos, ilist.begin(), ilist.end());
    }

    template<class... Args>
    __AGENCY_ANNOTATION
    iterator emplace(const_iterator pos, Args&&... args)
    {
      agency::sequenced_execution_policy seq;
      return emplace_n(seq, pos, 1, agency::detail::make_forwarding_iterator<Args&&>(&args)...);
    }

    __AGENCY_ANNOTATION
    iterator erase(const_iterator pos)
    {
      return erase(pos, pos + 1);
    }

    // XXX this needs an ExecutionPolicy overload
    __AGENCY_ANNOTATION
    iterator erase(const_iterator first_, const_iterator last_)
    {
      // get mutable iterators
      iterator first = begin() + (first_ - begin());
      iterator last = begin() + (last_ - begin());

      // overlap copy the range [last,end()) to first
      iterator old_end = end();
      end_ = agency::detail::overlapped_copy(last, end(), first);

      // destroy everything after end()
      agency::detail::destroy(storage_.allocator(), end(), old_end);

      // return an iterator referring to one past the last erased element
      return first;
    }

    __AGENCY_ANNOTATION
    void push_back(const T& value)
    {
      emplace_back(value);
    }

    __AGENCY_ANNOTATION
    void push_back(T&& value)
    {
      emplace_back(std::move(value));
    }

    template<class... Args>
    __AGENCY_ANNOTATION
    reference emplace_back(Args&&... args)
    {
      return *emplace(end(), std::forward<Args>(args)...);
    }

    __AGENCY_ANNOTATION
    void pop_back()
    {
      erase(end()-1, end());
    }

    // XXX this needs an ExecutionPolicy overload
    __AGENCY_ANNOTATION
    void resize(size_type new_size)
    {
      if(new_size < size())
      {
        agency::detail::destroy(begin() + new_size, end());
        end_ = begin() + new_size;
      }
      else
      {
        insert(end(), new_size - size(), T());
      }
    }

    // XXX this needs an ExecutionPolicy overload
    __AGENCY_ANNOTATION
    void resize(size_type new_size, const value_type& value)
    {
      if(new_size < size())
      {
        agency::detail::destroy(begin() + new_size, end());
        end_ = begin() + new_size;
      }
      else
      {
        insert(end(), new_size - size(), value);
      }
    }

    __AGENCY_ANNOTATION
    void swap(vector& other)
    {
      storage_.swap(other.storage_);
      agency::detail::adl_swap(end_, other.end_);
    }

  private:
    template<class ExecutionPolicy, class... InputIterator>
    __AGENCY_ANNOTATION
    iterator emplace_n(ExecutionPolicy&& policy, const_iterator position_, size_type count, InputIterator... iters)
    {
      // convert the const_iterator to an iterator
      iterator position = begin() + (position_ - cbegin());
      iterator result = position;

      if(count <= (capacity() - size()))
      {
        // we've got room for all of the new elements
        // how many existing elements will we displace?
        size_type num_displaced_elements = end() - position;
        iterator old_end = end();

        if(num_displaced_elements > count)
        {
          // move n displaced elements to newly constructed elements following the insertion
          end_ = agency::detail::uninitialized_move_n(policy, storage_.allocator(), end() - count, count, end());

          // copy construct num_displaced_elements - n elements to existing elements
          // this copy overlaps
          size_type copy_length = (old_end - count) - position;
          agency::detail::overlapped_uninitialized_copy(policy, storage_.allocator(), position, old_end - count, old_end - copy_length);

          // XXX we should destroy the elements [position, position + num_displaced_elements) before constructing new ones

          // construct new elements at insertion point
          agency::detail::construct_n(policy, storage_.allocator(), position, count, iters...);
        }
        else
        {
          // move already existing, displaced elements to the end of the emplaced range, which is at position + count
          end_ = agency::detail::uninitialized_move_n(policy, storage_.allocator(), position, num_displaced_elements, position + count);

          // XXX we should destroy the elements [position, position + num_displaced_elements) before placement newing new ones

          // construct new elements at the emplacement position
          agency::detail::construct_n(policy, storage_.allocator(), position, count, iters...);
        }
      }
      else
      {
        size_type old_size = size();

        // compute the new capacity after the allocation
        size_type new_capacity = old_size + agency::detail::max(old_size, count);

        // allocate exponentially larger new storage
        new_capacity = agency::detail::max(new_capacity, size_type(2) * capacity());

        // do not exceed maximum storage
        new_capacity = agency::detail::min(new_capacity, max_size());

        if(new_capacity > max_size())
        {
          detail::throw_length_error("insert(): insertion exceeds max_size().");
        }

        storage_type new_storage(new_capacity, storage_.allocator());

        // record how many constructors we invoke in the try block below
        iterator new_end = new_storage.data();

#ifndef __CUDA_ARCH__
        try
#endif
        {
          // move elements before the insertion to the beginning of the new storage
          new_end = agency::detail::uninitialized_move_n(policy, new_storage.allocator(), begin(), position - begin(), new_storage.data());

          result = new_end;

          // copy construct new elements
          new_end = agency::detail::construct_n(policy, new_storage.allocator(), new_end, count, iters...);

          // move elements after the insertion to the end of the new storage
          new_end = agency::detail::uninitialized_move_n(policy, new_storage.allocator(), position, end() - position, new_end);
        }
#ifndef __CUDA_ARCH__
        catch(...)
        {
          // something went wrong, so destroy as many new elements as were constructed
          agency::detail::destroy(policy, new_storage.allocator(), new_storage.data(), new_end);

          // rethrow
          throw;
        }
#endif

        // record the vector's new state
        storage_.swap(new_storage);
        end_ = new_end;
      }

      return result;
    }

    storage_type storage_;
    iterator end_;
};


template<class T, class Allocator>
__AGENCY_ANNOTATION
bool operator==(const vector<T,Allocator>& lhs, const vector<T,Allocator>& rhs)
{
  return lhs.size() == rhs.size() && agency::detail::equal(lhs.begin(), lhs.end(), rhs.begin());
}


template<class T, class Allocator>
__AGENCY_ANNOTATION
bool operator!=(const vector<T,Allocator>& lhs, const vector<T,Allocator>& rhs)
{
  return !(lhs == rhs);
}


// TODO
template<class T, class Allocator>
__AGENCY_ANNOTATION
bool operator<(const vector<T,Allocator>& lhs, const vector<T,Allocator>& rhs);

// TODO
template<class T, class Allocator>
__AGENCY_ANNOTATION
bool operator<=(const vector<T,Allocator>& lhs, const vector<T,Allocator>& rhs);

// TODO
template<class T, class Allocator>
__AGENCY_ANNOTATION
bool operator>(const vector<T,Allocator>& lhs, const vector<T,Allocator>& rhs);

// TODO
template<class T, class Allocator>
__AGENCY_ANNOTATION
bool operator>=(const vector<T,Allocator>& lhs, const vector<T,Allocator>& rhs);


template<class T, class Allocator>
__AGENCY_ANNOTATION
void swap(vector<T,Allocator>& a, vector<T,Allocator>& b)
{
  a.swap(b);
}


} // end experimental
} // end agency

