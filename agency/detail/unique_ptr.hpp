#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/swap.hpp>
#include <utility>
#include <memory>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class Allocator>
class deleter
{
  public:
    using value_type = typename std::allocator_traits<Allocator>::value_type;

    using pointer = typename std::allocator_traits<Allocator>::pointer;

    __AGENCY_ANNOTATION
    deleter() = default;

    __AGENCY_ANNOTATION
    deleter(const deleter&) = default;

    template<class OtherAllocator,
             class = typename std::enable_if<
               std::is_convertible<
                 typename std::allocator_traits<OtherAllocator>::pointer,
                 pointer
               >::value
             >::type
            >
    __AGENCY_ANNOTATION
    deleter(const deleter<OtherAllocator>&) {}

    __agency_hd_warning_disable__
    __AGENCY_ANNOTATION
    void operator()(pointer ptr) const
    {
      // destroy the object
      // XXX should use allocator_traits::destroy()
      ptr->~value_type();

      // deallocate
      Allocator alloc;
      alloc.deallocate(ptr, 1);
    }
};


template<class T>
using default_delete = deleter<std::allocator<T>>;


template<class T, class Deleter = default_delete<T>>
class unique_ptr
{
  public:
    using element_type = agency::detail::decay_t<T>;
    using pointer      = element_type*;
    using deleter_type = Deleter;

    __AGENCY_ANNOTATION
    unique_ptr(pointer ptr, const deleter_type& deleter = deleter_type())
      : ptr_(ptr),
        deleter_(deleter)
    {}

    __AGENCY_ANNOTATION
    unique_ptr() : unique_ptr(nullptr) {}
  
    __AGENCY_ANNOTATION
    unique_ptr(unique_ptr&& other)
      : ptr_(),
        deleter_(std::move(other.get_deleter()))
    {
      agency::detail::swap(ptr_, other.ptr_);
      agency::detail::swap(deleter_, other.deleter_);
    }

    template<class OtherT,
             class OtherDelete,
             class = typename std::enable_if<
               std::is_convertible<typename unique_ptr<OtherT,OtherDelete>::pointer,pointer>::value
             >::type
            >
    __AGENCY_ANNOTATION
    unique_ptr(unique_ptr<OtherT,OtherDelete>&& other)
      : ptr_(other.release()),
        deleter_(std::move(other.get_deleter()))
    {}
  
    __AGENCY_ANNOTATION
    ~unique_ptr()
    {
      reset(nullptr);
    }

    __AGENCY_ANNOTATION
    unique_ptr& operator=(unique_ptr&& other)
    {
      using agency::detail::swap;
      swap(ptr_,     other.ptr_);
      swap(deleter_, other.deleter_);
      return *this;
    }

    __AGENCY_ANNOTATION
    pointer get() const
    {
      return ptr_;
    }

    __AGENCY_ANNOTATION
    pointer release()
    {
      pointer result = nullptr;

      using agency::detail::swap;
      swap(ptr_, result);

      return result;
    }

    __AGENCY_ANNOTATION
    void reset(pointer ptr = pointer())
    {
      pointer old_ptr = ptr_;
      ptr_ = ptr;

      if(old_ptr != nullptr)
      {
        get_deleter()(old_ptr); 
      }
    }

    __AGENCY_ANNOTATION
    deleter_type& get_deleter()
    {
      return deleter_;
    }

    __AGENCY_ANNOTATION
    const deleter_type& get_deleter() const
    {
      return deleter_;
    }

    __AGENCY_ANNOTATION
    const T& operator*() const
    {
      return *ptr_;
    }

    __AGENCY_ANNOTATION
    T& operator*()
    {
      return *ptr_;
    }

    __AGENCY_ANNOTATION
    operator bool () const
    {
      return get();
    }

    __AGENCY_ANNOTATION
    void swap(unique_ptr& other)
    {
      agency::detail::swap(ptr_, other.ptr_);
      agency::detail::swap(deleter_, other.deleter_);
    }

  private:
    T* ptr_;
    deleter_type deleter_;
};


template<class T, class Alloc, class... Args>
__AGENCY_ANNOTATION
unique_ptr<T,deleter<Alloc>> allocate_unique(const Alloc& alloc, Args&&... args)
{
  Alloc alloc_copy = alloc;

  unique_ptr<T,deleter<Alloc>> result(alloc_copy.allocate(1), deleter<Alloc>());

  // XXX should use allocator_traits::construct()
  alloc_copy.template construct<T>(result.get(), std::forward<Args>(args)...);

  return std::move(result);
}


} // end detail
} // end agency

