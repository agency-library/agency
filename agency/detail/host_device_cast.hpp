#pragma once

#include <agency/detail/config.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class Function>
struct host_device_function
{
  mutable Function f_;

  __agency_hd_warning_disable__
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args) const
    -> decltype(
         f_(std::forward<Args>(args)...)
       )
  {
    return f_(std::forward<Args>(args)...);
  }
};


template<class Function>
__AGENCY_ANNOTATION
host_device_function<Function> host_device_cast(Function f)
{
  return host_device_function<Function>{f};
}


// avoid nested wrapping
template<class Function>
__AGENCY_ANNOTATION
host_device_function<Function> host_device_cast(host_device_function<Function> f)
{
  return f;
}


} // end detail
} // end agency

