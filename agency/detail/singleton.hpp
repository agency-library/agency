#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/terminate.hpp>


namespace agency
{
namespace detail
{


// this is a container for a value which takes an external flag
// tracking whether or not the value has been destroyed
// when the container is destroyed, the flag is set to indicate
// the object's lifetime has ended
template<class T>
class intrusive_optional
{
  public:
    intrusive_optional(bool& has_value) : has_value_(has_value)
    {
      has_value_ = true;
    }

    ~intrusive_optional()
    {
      has_value_ = false;
    }

    T& value()
    {
      return value_;
    }

  private:
    T value_;
    bool& has_value_;
};


template<class T>
__AGENCY_ANNOTATION
inline T* singleton()
{
#ifndef __CUDA_ARCH__
  // singleton() may be called after static destructors have completed
  // so the object by resource may no longer exist. track its lifetime with intrusive_optional
  // this is also why we return a pointer instead of a reference
  static bool has_value = false;
  static intrusive_optional<T> resource(has_value);

  return has_value ? &resource.value() : nullptr;
#else
  agency::cuda::detail::terminate_with_message("singleton(): This function is undefined in __device__ code.");
  return nullptr;
#endif
}


} // end detail
} // end agency

