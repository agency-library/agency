#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/utility.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/detail/allocation_deleter.hpp>
#include <utility>
#include <memory>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class T>
class default_delete : public allocation_deleter<std::allocator<T>>
{
  private:
    using super_t = allocation_deleter<std::allocator<T>>;

  public:
    using super_t::super_t;

    default_delete()
      : super_t(std::allocator<T>())
    {}
};


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
      agency::detail::adl_swap(ptr_, other.ptr_);
      agency::detail::adl_swap(deleter_, other.deleter_);
    }

    template<class OtherT,
             class OtherDeleter,
             class = typename std::enable_if<
               std::is_convertible<typename unique_ptr<OtherT,OtherDeleter>::pointer,pointer>::value
             >::type,
             class = typename std::enable_if<
               std::is_convertible<OtherDeleter&&, Deleter>::value
             >::type
            >
    __AGENCY_ANNOTATION
    unique_ptr(unique_ptr<OtherT,OtherDeleter>&& other)
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
      detail::adl_swap(ptr_,     other.ptr_);
      detail::adl_swap(deleter_, other.deleter_);
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

      agency::detail::adl_swap(ptr_, result);

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
      agency::detail::adl_swap(ptr_, other.ptr_);
      agency::detail::adl_swap(deleter_, other.deleter_);
    }

  private:
    T* ptr_;
    deleter_type deleter_;
};


__agency_exec_check_disable__
template<class T, class Alloc, class Deleter, class... Args>
__AGENCY_ANNOTATION
unique_ptr<T,Deleter> allocate_unique_with_deleter(const Alloc& alloc, const Deleter& deleter, Args&&... args)
{
  using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
  allocator_type alloc_copy = alloc;
  Deleter deleter_copy = deleter;

  unique_ptr<T,Deleter> result(alloc_copy.allocate(1), deleter_copy);

  allocator_traits<allocator_type>::construct(alloc_copy, result.get(), std::forward<Args>(args)...);

  return std::move(result);
}


template<class T, class Alloc, class... Args>
__AGENCY_ANNOTATION
unique_ptr<T,allocation_deleter<Alloc>> allocate_unique(const Alloc& alloc, Args&&... args)
{
  // because allocation_deleter<Alloc> derives T from its Alloc parameter,
  // we need to rebind Alloc to T before giving it to deleter
  using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;

  return allocate_unique_with_deleter<T>(alloc, allocation_deleter<allocator_type>(alloc), std::forward<Args>(args)...);
}


} // end detail
} // end agency

