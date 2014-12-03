#pragma once

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
__host__ __device__
void workaround_unused_variable_warning(const T&) {}


} // end detail
} // end cuda
} // end agency

