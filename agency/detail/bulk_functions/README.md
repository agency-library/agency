This directory contains the implementations of the bulk invocation functions:

  * `bulk_invoke`
  * `bulk_async`
  * `bulk_then`

The implementations of these functions share a similar structure.

The publically accessable entry points to these functions in `agency::` first use an `enable_if`
to guard the entry point and then lower on to functions in `agency::detail::`. For example,
`agency::bulk_invoke()` lowers onto `agency::detail::bulk_invoke_execution_policy()`.

`agency::detail::bulk_invoke_execution_policy()` sets up a function which
creates an execution agent and calls the user-provided function `f` with the agent
and other user-provided arguments as parameters to function `f`. This function is then invoked
using the lower-level `agency::detail::bulk_invoke_executor()` function.

