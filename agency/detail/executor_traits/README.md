This directory contains the implementations of `executor_traits`' member functions.

Each member function's implementation is selected from a set of
implementation strategies. The selection is made depending on the existence of
executor member functions and other criteria known at compile time.

Each implementation strategy is represented in the type system using an empty
tag type. For a particular `executor_traits` member function overload, these
tags are defined in the namespace
`agency::detail::`*function-overload-name*`_implementation_strategies`

The compile-time metafunction used to select the implementation strategy for a
particular `executor_traits` member function is named
`select_`*function-overload-name*`_implementation`. The result of this
metafunction is a tag type selected from the corresponding namespace.

The selected implementation of each executor_traits operation is invoked via tag-based dispatch.

