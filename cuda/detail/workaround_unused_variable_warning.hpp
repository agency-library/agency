#pragma once

namespace cuda
{
namespace detail
{


template<class T>
__host__ __device__
void workaround_unused_variable_warning(const T&) {}


}
}

