#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/detail/utility/blocking_bulk_twoway_execute.hpp>
#include <agency/execution/executor/detail/utility/bulk_async_execute_with_one_shared_parameter.hpp>
#include <agency/execution/executor/detail/utility/bulk_async_execute_without_shared_parameters.hpp>
#include <agency/execution/executor/detail/utility/bulk_async_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_async_execute_with_collected_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_continuation_executor_adaptor.hpp>
#include <agency/execution/executor/detail/utility/bulk_sync_execute_with_auto_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_sync_execute_with_auto_result_and_without_shared_parameters.hpp>
#include <agency/execution/executor/detail/utility/bulk_sync_execute_with_collected_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_sync_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_share_future.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_with_auto_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_with_collected_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_without_shared_parameters.hpp>
#include <agency/execution/executor/detail/utility/invoke_functors.hpp>

