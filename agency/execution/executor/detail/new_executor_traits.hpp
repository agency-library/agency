#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/detail/new_executor_traits/async_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute_with_one_shared_parameter.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute_without_shared_parameters.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_execute_with_auto_result.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_execute_with_auto_result_and_without_shared_parameters.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute_with_auto_result.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute_without_shared_parameters.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_continuation_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_synchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_continuation_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_synchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/sync_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/then_execute.hpp>

