#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <mutex>
#include <utility>
#include <memory>
#include <map>
#include <cassert>

namespace agency
{
namespace detail
{


template<class Alloc>
struct caching_memory_resource
  : private std::allocator_traits<Alloc>::template rebind_alloc<char>
{
  public:
    using base_allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<char>;

    __AGENCY_ANNOTATION
    caching_memory_resource() = default;

    __AGENCY_ANNOTATION
    caching_memory_resource(const caching_memory_resource&) = delete;

    __AGENCY_ANNOTATION
    caching_memory_resource(caching_memory_resource&&) = default;

    __AGENCY_ANNOTATION
    ~caching_memory_resource()
    {
#ifndef __CUDA_ARCH__
      deallocate_free_blocks();
#endif
    }

    using base_allocator_type::construct;

    __AGENCY_ANNOTATION
    base_allocator_type& get_allocator()
    {
      return *this;
    }

    __AGENCY_ANNOTATION
    const base_allocator_type& get_allocator() const
    {
      return *this;
    }

    __AGENCY_ANNOTATION
    void* allocate(size_t num_bytes)
    {
      char* ptr = nullptr;

#ifndef __CUDA_ARCH__
      std::lock_guard<std::mutex> guard(mutex_);

      // XXX this searches for a block of exactly the right size, but
      //     in general we could look for anything that fits, which is just
      //     the textbook malloc implementation
      auto free_block = free_blocks_.find(num_bytes);
      if(free_block != free_blocks_.end())
      {
        ptr = free_block->second;

        // erase from the free blocks map
        free_blocks_.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new allocation with the base allocator
        ptr = base_allocator_type::allocate(num_bytes);
      }

      // insert the allocation into the allocated_blocks map
      allocated_blocks_.insert(std::make_pair(ptr, num_bytes));
#else
      ptr = base_allocator_type::allocate(num_bytes);
#endif

      return ptr;
    }

    __AGENCY_ANNOTATION
    void deallocate(void* ptr, size_t num_bytes)
    {
#ifndef __CUDA_ARCH__
      std::lock_guard<std::mutex> guard(mutex_);

      // erase the allocation from the allocated blocks map
      auto found = allocated_blocks_.find(reinterpret_cast<char*>(ptr));
      assert(found != allocated_blocks_.end());
      allocated_blocks_.erase(found);

      // insert the block into the free blocks map
      free_blocks_.insert(std::make_pair(num_bytes, reinterpret_cast<char*>(ptr)));
#else
      base_allocator_type::deallocate(reinterpret_cast<char*>(ptr),num_bytes);
#endif
    }

  private:
    using free_blocks_type = std::multimap<size_t, char*>;
    using allocated_blocks_type = std::map<char*, size_t>;
    
    std::mutex            mutex_;
    free_blocks_type      free_blocks_;
    allocated_blocks_type allocated_blocks_;

    void deallocate_free_blocks()
    {
      std::lock_guard<std::mutex> guard(mutex_);

      for(auto b : free_blocks_)
      {
        base_allocator_type::deallocate(b.second, b.first);
      }
      free_blocks_.clear();

      // note that we do not attempt to deallocate allocated blocks
      // that's the user's responsibility
    }
}; // end caching_memory_resource


template<class Resource>
__AGENCY_ANNOTATION
Resource* get_system_resource()
{
#ifndef __CUDA_ARCH__
  static Resource resource;
  return &resource;
#else
  agency::cuda::detail::terminate_with_message("get_system_resource(): This function is undefined in __device__ code.");
  return nullptr;
#endif
}


template<class Alloc>
class caching_allocator
{
  public:
    __AGENCY_ANNOTATION
    caching_allocator()
      : resource_(*get_system_resource<resource_type>())
    {}

    using value_type = typename std::allocator_traits<Alloc>::value_type;
    using pointer    = typename std::allocator_traits<Alloc>::pointer;
    using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
    using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
    using const_void_pointer = typename std::allocator_traits<Alloc>::const_void_pointer;
    using difference_type = typename std::allocator_traits<Alloc>::difference_type;
    using size_type = typename std::allocator_traits<Alloc>::size_type;

    using propagate_on_container_copy_assignment = typename std::allocator_traits<Alloc>::propagate_on_container_copy_assignment;
    using propagate_on_container_move_assignment = typename std::allocator_traits<Alloc>::propagate_on_container_move_assignment;
    using propagate_on_container_swap = typename std::allocator_traits<Alloc>::propagate_on_container_swap;

    template<class T>
    struct rebind
    {
      using other = caching_allocator<
        typename std::allocator_traits<Alloc>::template rebind_alloc<T>
      >;
    };

    __AGENCY_ANNOTATION
    pointer allocate(size_type n)
    {
      return reinterpret_cast<pointer>(resource_.allocate(n * sizeof(value_type)));
    }

    __AGENCY_ANNOTATION
    void deallocate(pointer ptr, size_type n)
    {
      resource_.deallocate(ptr, n * sizeof(value_type));
    }

    __agency_hd_warning_disable__
    template<class T, class... Args>
    __AGENCY_ANNOTATION
    void construct(T* ptr, Args&&... args)
    {
      return resource_.get_allocator().construct(ptr, std::forward<Args>(args)...);
    }

    __agency_hd_warning_disable__
    template<class T>
    __AGENCY_ANNOTATION
    void destroy(T* ptr, size_type n)
    {
      resource_.get_allocator().destroy(ptr, n);
    }

  private:
    using resource_type = caching_memory_resource<
      typename std::allocator_traits<Alloc>::template rebind_alloc<char>
    >;

    resource_type& resource_;
};


} // end detail
} // end agency

