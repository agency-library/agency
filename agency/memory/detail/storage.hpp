#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>

namespace agency
{
namespace detail
{


// XXX move this underneath container/detail/storage.hpp
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
        data_ = detail::allocator_traits<Allocator>::allocate(allocator_, count);
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
      detail::allocator_traits<Allocator>::deallocate(allocator_, data(), size());
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
      detail::adl_swap(data_, other.data_);
      detail::adl_swap(size_, other.size_);

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
      detail::adl_swap(data_, other.data_);
      detail::adl_swap(size_, other.size_);
      detail::adl_swap(allocator_, other.allocator_);
    }

  private:
    T* data_;
    size_t size_;
    Allocator allocator_;
};

} // end detail
} // end agency

