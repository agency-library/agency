This directory contains functions which provide the semantics of one execution function by adapting some other execution function provided natively by a given executor.

For example, the function `bulk_then_execute_via_bulk_twoway_execute` adapts an executor's native behavior provided via its `.bulk_twoway_execute` member function to deliver the semantics of `bulk_then_execute`.

