This directory contains the implementations of `executor_traits`' member functions.

Each member function's implementation is selected from a set of
implementation strategies. The selection is made depending on the existence of
executor member functions and other criteria known at compile time.

Each implementation strategy is represented in the type system using an empty
tag type. For a particular `executor_traits` member function overload, these
tags are defined in the namespace
`agency::detail::`*function-overload-name*`_implementation_strategies`. The order in which these types are defined is also the priority with which the corresponding implementation strategies are selected.

The compile-time metafunction used to select the implementation strategy for a
particular `executor_traits` member function is named
`select_`*function-overload-name*`_implementation`. The result of this
metafunction is a tag type selected from the corresponding namespace.

The selected implementation of each `executor_traits` member function is invoked via tag-based dispatch.

Most implementation strategies first attempt to call the operation of interest via `ex.`*function-overload-name*`(args)`. If the member function of interest does not exist, other member functions may be tried next. Failing that, an implementation will recurse through an operation called via `executor_traits`, whose invocation always succeeds.

`then_execute` is the terminal operation -- its implementation does not attempt to continue recursion through `executor_traits`.
