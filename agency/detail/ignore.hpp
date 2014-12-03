#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{


struct ignore_t
{
  template<class T>
  __AGENCY_ANNOTATION
  const ignore_t operator=(T&&) const
  {
    return *this;
  }
};

namespace
{


constexpr ignore_t ignore{};


} // end anon namespace


} // end detail
} // end agency

