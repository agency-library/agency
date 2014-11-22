#pragma once

#include "grid_executor.hpp"
#include <agency/flattened_executor.hpp>

namespace cuda
{


using parallel_executor = agency::flattened_executor<cuda::grid_executor>;


}

