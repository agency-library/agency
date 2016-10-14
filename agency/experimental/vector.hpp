#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
#include <agency/detail/utility.hpp>
#include <memory>

namespace agency
{
namespace experimental
{
namespace detail
{


// XXX all of these uninitialized_* algorithms ought to lower onto construct_each()
// maybe construct_each() could look like this:
//
// template<class Iterator, class... Iterators>
// void construct_each(allocator_type& alloc, Iterator first, Iterator last, Iterators... iters)
// {
//   for(; first != last; ++first, ++iters...)
//   {
//     allocator_traits<Allocator>::construct(alloc, *first, *iters...);
//   }
// }

template<class Allocator, class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
Iterator2 uninitialized_move(Allocator& alloc, Iterator1 first, Iterator1 last, Iterator2 result)
{
  for(; first != last; ++first, ++result)
  {
    agency::detail::allocator_traits<Allocator>::construct(alloc, &*result, std::move(*first));
  }

  return result;
}


template<class Allocator, class Iterator, class Size, class T>
__AGENCY_ANNOTATION
Iterator uninitialized_fill_n(Allocator& alloc, Iterator first, Size n, const T& value)
{
  for(Size i = 0; i < n; ++i, ++first)
  {
    agency::detail::allocator_traits<Allocator>::construct(alloc, &*first, value);
  }

  return first;
}


template<class Allocator, class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
Iterator2 uninitialized_copy(Allocator& alloc, Iterator1 first, Iterator1 last, Iterator2 result)
{
  for(; first != last; ++first, ++result)
  {
    agency::detail::allocator_traits<Allocator>::construct(alloc, &*result, *first);
  }

  return result;
}


template<class Allocator, class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
Iterator2 uninitialized_copy_backward(Allocator& alloc, Iterator1 first, Iterator1 last, Iterator2 result)
{
  // yes, we preincrement
  // the ranges are open on the right, i.e. [first, last)
  while(first != last)
  {
    agency::detail::allocator_traits<Allocator>::construct(alloc, &*--result, *--last);
  }

  return result;
}


template<class Allocator, class Iterator>
__AGENCY_ANNOTATION
Iterator overlapped_uninitialized_copy(Allocator& alloc, Iterator first, Iterator last, Iterator result)
{
  if(first < last && first <= result && result < last)
  {
    // result lies in [first, last)
    // it's safe to use uninitialized_copy_backward here
    detail::uninitialized_copy_backward(alloc, first, last, result + (last - first));
    result += (last - first);
  }
  else
  {
    // result + (last - first) lies in [first, last)
    // it's safe to use uninitialized_copy here
    result = detail::uninitialized_copy(alloc, first, last, result);
  } // end else

  return result;
}


// XXX seems like this should be a member of allocator_traits

template<class Allocator, class Iterator>
__AGENCY_ANNOTATION
void destroy_each(Allocator& alloc, Iterator first, Iterator last)
{
  for(; first != last; ++first)
  {
    agency::detail::allocator_traits<Allocator>::destroy(alloc, &*first);
  }
}


__AGENCY_ANNOTATION
inline void throw_bad_alloc()
{
#ifdef __CUDA_ARCH__
  printf("bad_alloc");
  assert(0);
#else
  throw std::bad_alloc();
#endif
}


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


template<class T, class Allocator>
class storage
{
  public:
    __AGENCY_ANNOTATION
    storage(size_t count, const Allocator& allocator)
      : data_(nullptr),
        size_(count),
        allocator_(allocator)
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
    storage(const Allocator& allocator)
      : storage(0, allocator)
    {}

    __AGENCY_ANNOTATION
    storage()
      : storage(Allocator())
    {}

    __AGENCY_ANNOTATION
    ~storage()
    {
      allocator_.deallocate(data(), size());
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


template<class T>
class constant_iterator
{
  public:
    using value_type = T;
    using reference = const value_type&;
    using pointer = const value_type*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    __AGENCY_ANNOTATION
    constant_iterator(const T& value, size_t position)
      : value_(value), position_(position)
    {}

    __AGENCY_ANNOTATION
    constant_iterator(const T& value)
      : constant_iterator(value, 0)
    {}

    // dereference
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return value_;
    }

    // not equal
    __AGENCY_ANNOTATION
    bool operator!=(const constant_iterator& rhs) const
    {
      return position_ != rhs.position_;
    }

    // pre-increment
    __AGENCY_ANNOTATION
    constant_iterator& operator++()
    {
      ++position_;
      return *this;
    }

    // pre-decrement
    __AGENCY_ANNOTATION
    constant_iterator& operator--()
    {
      --position_;
      return *this;
    }

    // plus-equal
    __AGENCY_ANNOTATION
    constant_iterator& operator+=(difference_type n)
    {
      position_ += n;
      return *this;
    }

    // difference
    __AGENCY_ANNOTATION
    difference_type operator-(const constant_iterator& rhs)
    {
      return position_ - rhs.position_;
    }

  private:
    T value_;
    size_t position_;
};


} // end detail


template<class T, class Allocator = std::allocator<T>>
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

    __AGENCY_ANNOTATION
    vector()
      : storage_(), end_(nullptr)
    {}

    __AGENCY_ANNOTATION
    vector(size_type count, const T& value, const Allocator& alloc = Allocator())
      : storage_(alloc), end_(nullptr)
    {
      insert(end(), count, value);
    }

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
        end_ = detail::uninitialized_copy(new_storage.allocator(), begin(), end(), new_storage.data());

        // swap out our storage
        storage_.swap(new_storage);
      }
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

    __AGENCY_ANNOTATION
    size_type capacity() const
    {
      return storage_.size();
    }

    __AGENCY_ANNOTATION
    iterator begin()
    {
      return storage_.data();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return end_;
    }

    __AGENCY_ANNOTATION
    const_iterator cbegin() const
    {
      return storage_.data();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return cbegin();
    }

    __AGENCY_ANNOTATION
    const_iterator cend() const
    {
      return end_;
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return cend();
    }

    __AGENCY_ANNOTATION
    bool empty() const
    {
      return cbegin() == cend();
    }

    __AGENCY_ANNOTATION
    iterator insert(const_iterator position, size_type count, const T& value)
    {
      return insert(position, detail::constant_iterator<T>(value, 0), detail::constant_iterator<T>(value, count));
    }

    // XXX should be able to relax this to work with all iterators
    template<class RandomAccessIterator,
             __AGENCY_REQUIRES(
               std::is_convertible<
                 typename std::iterator_traits<RandomAccessIterator>::iterator_category,
                 std::random_access_iterator_tag
               >::value
             )
            >
    __AGENCY_ANNOTATION
    iterator insert(const_iterator position_, RandomAccessIterator first, RandomAccessIterator last)
    {
      // convert the const_iterator to an iterator
      iterator position = begin() + (position_ - cbegin());
      iterator result = position;

      size_type count = last - first;

      if(count <= (capacity() - size()))
      {
        // we've got room for all of the new elements
        // how many existing elements will we displace?
        size_type num_displaced_elements = end() - position;
        iterator old_end = end();

        if(num_displaced_elements > count)
        {
          // move n displaced elements to newly constructed elements following the insertion
          end_ = detail::uninitialized_move(storage_.allocator(), end() - count, end(), end());

          // copy construct num_displaced_elements - n elements to existing elements
          // this copy overlaps
          size_type copy_length = (old_end - count) - position;
          detail::overlapped_uninitialized_copy(storage_.allocator(), position, old_end - count, old_end - copy_length);

          // finally, copy the range to the insertion point
          detail::uninitialized_copy(storage_.allocator(), first, last, position);
        }
        else
        {
          RandomAccessIterator mid = first;
          mid += num_displaced_elements;

          // construct copy new elements at the end of the vector
          end_ = detail::uninitialized_copy(storage_.allocator(), mid, last, end());

          // move the displaced elements
          end_ = detail::uninitialized_move(storage_.allocator(), position, old_end, end());

          // construct copy at the insertion position
          detail::uninitialized_copy(storage_.allocator(), first, mid, position);
        }
      }
      else
      {
        size_type old_size = size();

        // compute the new capacity after the allocation
        size_type new_capacity = old_size + std::max(old_size, count);

        // allocate exponentially larger new storage
        new_capacity = std::max(new_capacity, size_type(2) * capacity());

        // do not exceed maximum storage
        new_capacity = std::min(new_capacity, max_size());

        if(new_capacity > max_size())
        {
          detail::throw_length_error("insert(): insertion exceeds max_size().");
        }

        storage_type new_storage(new_capacity, storage_.allocator());

        // record how many constructors we invoke in the try block below
        iterator new_end = new_storage.data();

        try
        {
          // move elements before the insertion to the beginning of the new storage
          new_end = detail::uninitialized_move(new_storage.allocator(), begin(), position, new_storage.data());

          result = new_end;

          // copy construct new elements
          new_end = detail::uninitialized_copy(new_storage.allocator(), first, last, new_end);

          // move elements after the insertion to the end of the new storage
          new_end = detail::uninitialized_move(new_storage.allocator(), position, end(), new_end);
        }
        catch(...)
        {
          // something went wrong, so destroy as many new elements as were constructed
          detail::destroy_each(new_storage.allocator(), new_storage.data(), new_end);

          // rethrow
          throw;
        }

        // record the vector's new state
        storage_.swap(new_storage);
        end_ = new_end;
      }

      return result;
    }

  private:
    storage_type storage_;
    iterator end_;
};


} // end experimental
} // end agency

