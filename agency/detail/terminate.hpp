#pragma once

#include <agency/detail/config.hpp>
#include <exception>
#include <stdexcept>
#include <cstdio>


namespace agency
{
namespace detail
{


// XXX consider using __cpp_exceptions below to portably check for exception support
//     rather than using __CUDA_ARCH__


__AGENCY_ANNOTATION
inline void terminate()
{
#ifdef __CUDA_ARCH__
  asm("trap;");
#else
  std::terminate();
#endif
}


__AGENCY_ANNOTATION
inline void terminate_with_message(const char* message)
{
  printf("%s\n", message);

  detail::terminate();
}


__AGENCY_ANNOTATION
inline void throw_runtime_error(const char* message)
{
#ifndef __CUDA_ARCH__
  throw std::runtime_error(message);
#else
  detail::terminate_with_message(message);
#endif
}


} // end detail
} // end agency

