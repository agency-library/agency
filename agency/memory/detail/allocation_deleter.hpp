#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/utility.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>

namespace agency
{
namespace detail
{


// see https://wg21.link/P0211
// allocation_deleter is a deleter which uses an Allocator's .destroy() function to delete objects
template<class Allocator>
class allocation_deleter : private Allocator // use inheritance for empty base class optimization
{
  public:
    using pointer = typename std::allocator_traits<Allocator>::pointer;

    __AGENCY_ANNOTATION
    allocation_deleter()
      : allocation_deleter(Allocator())
    {}

    __AGENCY_ANNOTATION
    allocation_deleter(const Allocator& alloc)
      : Allocator(alloc)
    {}

    template<class OtherAllocator,
             __AGENCY_REQUIRES(
               std::is_convertible<typename std::allocator_traits<OtherAllocator>::pointer, pointer>::value
             )
            >
    __AGENCY_ANNOTATION
    allocation_deleter(const allocation_deleter<OtherAllocator>& other)
      : Allocator(other)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    void operator()(pointer ptr) const
    {
      // XXX P0211 suggests to use allocator_delete() here,
      //     but just do everything manually for now
      
      Allocator& alloc = const_cast<Allocator&>(static_cast<const Allocator&>(*this));

      // destroy the object
      allocator_traits<Allocator>::destroy(alloc, ptr);

      // deallocate
      allocator_traits<Allocator>::deallocate(alloc, ptr, 1);
    }

    __AGENCY_ANNOTATION
    void swap(allocation_deleter& other)
    {
      detail::adl_swap(static_cast<Allocator&>(*this), static_cast<Allocator&>(other));
    }

  private:
    // allocation_deleter's copy constructor needs access to all other allocation_deleters' (private) base class
    template<class OtherAllocator> friend class allocation_deleter;
};


template<class Allocator>
__AGENCY_ANNOTATION
void swap(allocation_deleter<Allocator>& a, allocation_deleter<Allocator>& b)
{
  a.swap(b);
}


} // end detail
} // end agency

