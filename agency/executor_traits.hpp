#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>


namespace agency
{


template<class Executor>
using executor_traits = new_executor_traits<Executor>;


} // end agency

