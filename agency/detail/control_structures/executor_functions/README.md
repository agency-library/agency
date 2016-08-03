This directory contains the implementations of lower-level forms of the bulk invocation functions:

  * `bulk_invoke_executor`
  * `bulk_async_executor`
  * `bulk_then_executor`

These work like the regular `bulk_invoke` etc. functions, but take an executor
as a parameter rather than an execution policy. The implementation of
higher-level forms of `bulk_invoke` and friends lower onto these functions.

