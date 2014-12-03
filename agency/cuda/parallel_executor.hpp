#pragma once

#include <agency/cuda/grid_executor.hpp>
#include <agency/flattened_executor.hpp>

namespace agency
{
namespace cuda
{


using parallel_executor = agency::flattened_executor<grid_executor>;


} // end cuda
} // end agency

