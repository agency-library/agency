#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/singleton.hpp>
#include <mutex>
#include <map>

namespace agency
{
namespace detail
{


template<class MemoryResource>
class cached_resource : private MemoryResource
{
  public:
    using resource_type = MemoryResource;

    // inherit the base resource's constructors
    using resource_type::resource_type;

    cached_resource() = default;

    cached_resource(const cached_resource&) = delete;

    cached_resource(cached_resource&&) = default;

    ~cached_resource()
    {
      deallocate_free_blocks();
    }

    void* allocate(size_t num_bytes)
    {
      void* ptr = nullptr;

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
        // create a new allocation with the base resource
        ptr = resource_type::allocate(num_bytes);
      }

      // insert the allocation into the allocated_blocks map
      allocated_blocks_.insert(std::make_pair(ptr, num_bytes));

      return ptr;
    }

    void deallocate(void* ptr, size_t num_bytes)
    {
      // erase the allocation from the allocated blocks map
      auto found = allocated_blocks_.find(reinterpret_cast<char*>(ptr));
      allocated_blocks_.erase(found);

      // insert the block into the free blocks map
      free_blocks_.insert(std::make_pair(num_bytes, reinterpret_cast<char*>(ptr)));
    }

    bool operator==(const cached_resource& other) const
    {
      return this == &other;
    }

    bool operator!=(const cached_resource& other) const
    {
      return this != &other;
    }

  private:
    using free_blocks_type = std::multimap<size_t, void*>;
    using allocated_blocks_type = std::map<void*, size_t>;
    
    free_blocks_type      free_blocks_;
    allocated_blocks_type allocated_blocks_;

    void deallocate_free_blocks()
    {
      for(auto b : free_blocks_)
      {
        // since this is only called from the destructor,
        // swallow any exceptions thrown by this call to
        // deallocate in order to avoid propagating exceptions
        // out of destructors
        try
        {
          resource_type::deallocate(b.second, b.first);
        }
        catch(...)
        {
          // just swallow any exceptions we encounter
        }
      }
      free_blocks_.clear();

      // note that we do not attempt to deallocate allocated blocks
      // that's the user's responsibility
    }
};


template<class MemoryResource>
struct cached_resources_singleton_t
{
  std::mutex mutex;
  std::map<MemoryResource, cached_resource<MemoryResource>> cached_resources;
};


template<class MemoryResource>
inline cached_resources_singleton_t<MemoryResource>* cached_resources_singleton()
{
  return agency::detail::singleton<cached_resources_singleton_t<MemoryResource>>();
}


template<class MemoryResource>
inline void* allocate_from_cached_resources_singleton(const MemoryResource& resource, size_t num_bytes)
{
  void* result = nullptr;

  cached_resources_singleton_t<MemoryResource>* resources_ptr = cached_resources_singleton<MemoryResource>();

  if(resources_ptr)
  {
    // lock the resources
    std::lock_guard<std::mutex> guard(resources_ptr->mutex);

    // allocate using the cached resource associated with the given resource
    result = resources_ptr->cached_resources[resource].allocate(num_bytes);
  }

  return result;
}


template<class MemoryResource>
inline void deallocate_from_cached_resources_singleton(const MemoryResource& resource, void* ptr, size_t num_bytes)
{
  cached_resources_singleton_t<MemoryResource>* resources_ptr = cached_resources_singleton<MemoryResource>();

  if(resources_ptr)
  {
    // lock the resources
    std::lock_guard<std::mutex> guard(resources_ptr->mutex);

    // deallocate using the cached resource associated with the given resource
    resources_ptr->cached_resources[resource].deallocate(ptr, num_bytes);
  }
}


template<class MemoryResource>
class globally_cached_resource
{
  public:
    globally_cached_resource(const MemoryResource& resource)
      : resource_(resource)
    {}

    globally_cached_resource()
      : globally_cached_resource(MemoryResource())
    {}

    globally_cached_resource(const globally_cached_resource&) = default;

    inline void* allocate(size_t num_bytes)
    {
      return allocate_from_cached_resources_singleton(resource_, num_bytes);
    }

    inline void deallocate(void *ptr, size_t num_bytes)
    {
      deallocate_from_cached_resources_singleton(resource_, ptr, num_bytes);
    }

    bool operator==(const globally_cached_resource& other) const
    {
      return resource_ == other.resource_;
    }

    bool operator!=(const globally_cached_resource& other) const
    {
      return resource_ != other.resource_;
    }

  private:
    MemoryResource resource_;
};


} // end detail
} // end agency

