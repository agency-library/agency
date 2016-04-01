#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/bulk_invoke/scope_result.hpp>


namespace agency
{


template<class T>
using single_result = scope_result<T,0>;


} // end agency

