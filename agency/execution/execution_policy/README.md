This directory contains definitions of Agency's built-in execution policy
types. These are all defined by inheriting from `basic_execution_policy`, using
the appropriate types as template parameters to describe their essential
characteristics. We use inheritance, instead of making a simple `typedef` of
`basic_execution_policy`, for the sake of better documentation and compiler error
messages. Defining a unique type like `sequenced_execution_policy` ensures that
"`sequenced_execution_policy`" appears in compiler output, instead of some type
recipe involving `basic_execution_policy`.

