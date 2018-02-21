This directory contains "execution functions":

  * `bulk_then_execute`
  * `bulk_twoway_execute`
  * `then_execute`
  * `twoway_execute`

These are free functions which receive an executor and other parameters, and create execution agents.

If the given executor natively supports the execution function via a member function, then the execution function simply calls that member function.

Otherwise, the executor's native support is adapted to provide the execution function's semantics.

