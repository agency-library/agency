#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/bulk_functions/scope_result.hpp>


namespace agency
{


template<class T>
using single_result = scope_result<0,T>;


} // end agency

