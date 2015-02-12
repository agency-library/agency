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


std::ostream& operator<<(std::ostream& os, ignore_t)
{
  return std::cout << "ignore";
}

namespace
{


constexpr ignore_t ignore{};


} // end anon namespace


} // end detail
} // end agency

