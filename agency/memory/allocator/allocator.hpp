#pragma once

#include <agency/detail/config.hpp>
#include <cstddef>
#include <type_traits>

namespace agency
{
namespace detail
{


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


} // end detail


template<class T>
struct allocator
{
  using value_type = T;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  allocator() = default;

  allocator(const allocator&) = default;

  template<class U>
  __AGENCY_ANNOTATION
  allocator(const allocator<U>&){}

  __AGENCY_ANNOTATION
  ~allocator(){}

  __AGENCY_ANNOTATION
  T* allocate(std::size_t n)
  {
    T* result = static_cast<T*>(::operator new(sizeof(value_type) * n));
    if(result == nullptr)
    {
      detail::throw_bad_alloc();
    }

    return result;
  }

  __AGENCY_ANNOTATION
  void deallocate(T* p, std::size_t)
  {
    ::operator delete(p);
  }
};


template<class T1, class T2>
__AGENCY_ANNOTATION
bool operator==(const allocator<T1>&, const allocator<T2>&)
{
  return true;
}


template<class T1, class T2>
__AGENCY_ANNOTATION
bool operator!=(const allocator<T1>&, const allocator<T2>&)
{
  return false;
}


} // end agency

