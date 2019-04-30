#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/cuda/memory/device_ptr.hpp>
//#include <agency/execution/execution_policy/basic_execution_policy.hpp>
//#include <agency/execution/execution_policy/parallel_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/parallel_execution_policy.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/functional/invoke.hpp>
#include <agency/cuda/device.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#include <agency/experimental/ndarray/shape.hpp>
#include <agency/experimental/ndarray/domain.hpp>

#include <cstdlib>
#include <utility>


namespace agency
{
namespace cuda
{
namespace detail
{
namespace device_allocator_detail
{


template<class T>
__AGENCY_ANNOTATION
constexpr T&& unwrap_device_reference(T&& r) noexcept
{
  return std::forward<T>(r);
}


template<class T>
__AGENCY_ANNOTATION
constexpr T& unwrap_device_reference(const agency::cuda::device_reference<T>& r) noexcept
{
  return *(&r).get();
}

template<class T>
__AGENCY_ANNOTATION
constexpr T& unwrap_device_reference(agency::cuda::device_reference<T>&& r) noexcept
{
  return *(&r).get();
}


struct bulk_construct_functor
{
  template<class Agent, class Allocator, class ArrayView, class... ArrayViews>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, Allocator alloc, ArrayView array, ArrayViews... arrays) const
  {
    // get the index of the array element to construct
    auto idx = self.index();

    // use single-object construct with the given allocator
    agency::detail::allocator_traits<Allocator>::construct(alloc, &array[idx], unwrap_device_reference(arrays[idx])...);
  }
};


struct bulk_destroy_functor
{
  template<class Agent, class Allocator, class ArrayView>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, Allocator alloc, ArrayView array) const
  {
    // get the index of the array element to construct
    auto idx = self.index();

    // use single-object destroy with the given allocator
    agency::detail::allocator_traits<Allocator>::destroy(alloc, &array[idx]);
  }
};


struct placement_new_functor
{
  template<class T, class... Args>
  __AGENCY_ANNOTATION
  void operator()(T* ptr, Args&&... args)
  {
    new(ptr) T(std::forward<Args>(args)...);
  }
};


struct destroy_functor
{
  template<class T>
  __AGENCY_ANNOTATION
  void operator()(T* ptr)
  {
    ptr->~T();
  }
};


} // end device_allocator_detail
} // end detail


template<class T>
class device_allocator
{
  public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = device_ptr<T>;
    using const_pointer = device_ptr<const T>;
    using reference = device_reference<T>;
    using const_reference = device_reference<const T>;

    __AGENCY_ANNOTATION
    explicit device_allocator(const parallel_executor& executor = parallel_executor())
      : executor_(executor)
    {}

    __AGENCY_ANNOTATION
    explicit device_allocator(const device_id& device)
      : device_allocator(parallel_executor(grid_executor(device)))
    {}

    __AGENCY_ANNOTATION
    device_allocator(const device_allocator& other)
      : executor_(other.executor_)
    {}

    template<class U>
    __AGENCY_ANNOTATION
    device_allocator(const device_allocator<U>& other)
      : executor_(other.executor())
    {}

    __AGENCY_ANNOTATION
    const parallel_executor& executor() const
    {
      return executor_;
    }

    __AGENCY_ANNOTATION
    pointer allocate(std::size_t n)
    {
#ifndef __CUDA_ARCH__
      // switch to our device
      scoped_device set_current_device(device());
#endif // __CUDA_ARCH__

      T* raw_ptr = nullptr;

      // allocate
#if __cuda_lib_has_cudart
      agency::cuda::detail::throw_on_error(cudaMalloc(reinterpret_cast<void**>(&raw_ptr), n * sizeof(T)), "device_allocator::allocate");
#else
      raw_ptr = static_cast<T*>(malloc(n * sizeof(T)));
#endif // __cuda_lib_has_cudart

      return pointer{raw_ptr};
    }

    __AGENCY_ANNOTATION
    void deallocate(pointer ptr, std::size_t)
    {
#ifndef __CUDA_ARCH__
      // switch to our device
      scoped_device set_current_device(device());
#endif // __CUDA_ARCH__

      // deallocate
#if __cuda_lib_has_cudart
      agency::cuda::detail::throw_on_error(cudaFree(ptr.get()), "device_allocator::deallocate");
#else
      free(ptr.get());
#endif // __cuda_lib_has_cudart
    }

    template<class U, class... Args>
    __AGENCY_ANNOTATION
    void construct(device_ptr<U> ptr, Args&&... args)
    {
      // this workaround ensures that both the host and device paths are instantiated,
      // even if they are not ultimately called when __CUDA_ARCH__ is used as a guard
      struct workaround
      {
        __host__
        static void host_path(device_allocator& self, device_ptr<U> ptr, Args&&... args)
        {
          // when called from the host, invoke placement new using our executor
          agency::invoke(
            self.executor_,
            detail::device_allocator_detail::placement_new_functor{},
            ptr.get(),
            args...
          );
        }

        __device__
        static void device_path(device_allocator&, device_ptr<U> ptr, Args&&... args)
        {
          // placement new into the raw pointer on the device
          new(ptr.get()) U(std::forward<Args>(args)...);
        }
      };

#ifndef __CUDA_ARCH__
      workaround::host_path(*this, ptr, std::forward<Args>(args)...);
#else
      workaround::device_path(*this, ptr, std::forward<Args>(args)...);
#endif // __CUDA_ARCH__
    }

    template<class ArrayView, class... ArrayViews>
    __AGENCY_ANNOTATION
    void bulk_construct(ArrayView array, ArrayViews... arrays)
    {
      // this workaround ensures that both the host and device paths are instantiated,
      // even if they are not ultimately called when __CUDA_ARCH__ is used as a guard
      struct workaround
      {
        __host__
        static void host_path(device_allocator& self, ArrayView array, ArrayViews... arrays)
        {
          auto shape = agency::experimental::shape(array);

          // when called from the host, bulk_invoke constructors on the device using our execution policy
          agency::bulk_invoke(
            cuda::parnd(shape).on(self.executor()),
            detail::device_allocator_detail::bulk_construct_functor{},
            self,
            array,
            arrays...
          );
        }

        __device__
        static void device_path(device_allocator& self, ArrayView array, ArrayViews... arrays)
        {
          // iterate through array's domain and call construct
          for(auto idx : agency::experimental::domain(array))
          {
            self.construct(&array[idx], detail::device_allocator_detail::unwrap_device_reference(arrays[idx])...);
          }
        }
      };

#ifndef __CUDA_ARCH__
      workaround::host_path(*this, array, arrays...);
#else
      workaround::device_path(*this, array, arrays...);
#endif // __CUDA_ARCH__
    }

    template<class U>
    __AGENCY_ANNOTATION
    void destroy(device_ptr<U> ptr)
    {
      // this workaround ensures that both the host and device paths are instantiated,k
      // even if they are not ultimately called when __CUDA_ARCH__ is used as a guard
      struct workaround
      {
        __host__
        static void host_path(device_allocator& self, device_ptr<U> ptr)
        {
          // when called from the host, invoke the destructor using our executor
          agency::invoke(
            self.executor_,
            detail::device_allocator_detail::destroy_functor{},
            ptr.get()
          );
        }

        __device__
        static void device_path(device_allocator&, device_ptr<U> ptr)
        {
          // call the destructor on the raw pointer
          ptr.get()->~U();
        }
      };

#ifndef __CUDA_ARCH__
      workaround::host_path(*this, ptr);
#else
      workaround::device_path(*this, ptr);
#endif // __CUDA_ARCH__
    }

    template<class ArrayView>
    __AGENCY_ANNOTATION
    void bulk_destroy(ArrayView array)
    {
      // this workaround ensures that both the host and device paths are instantiated,
      // even if they are not ultimately called when __CUDA_ARCH__ is used as a guard
      struct workaround
      {
        __host__
        static void host_path(device_allocator& self, ArrayView array)
        {
          auto shape = agency::experimental::shape(array);

          // when called from the host, bulk_invoke destructors on the device using our execution policy
          agency::bulk_invoke(
            cuda::parnd(shape).on(self.executor()),
            detail::device_allocator_detail::bulk_destroy_functor{},
            self,
            array
          );
        }

        __device__
        static void device_path(device_allocator& self, ArrayView array)
        {
          // destroy each array element
          for(auto idx : agency::experimental::domain(array))
          {
            self.destroy(&array[idx]);
          }
        }
      };

#ifndef __CUDA_ARCH__
      workaround::host_path(*this, array);
#else
      workaround::device_path(*this, array);
#endif // __CUDA_ARCH__
    }

    __AGENCY_ANNOTATION
    bool operator==(const device_allocator& other) const
    {
      return executor_ == other.executor_;
    }

    __AGENCY_ANNOTATION
    bool operator!=(const device_allocator& other) const
    {
      return !operator==(other);
    }

  private:
    __AGENCY_ANNOTATION
    device_id device() const
    {
      return executor_.base_executor().device();
    }

    parallel_executor executor_;
};


} // end cuda
} // end agency

