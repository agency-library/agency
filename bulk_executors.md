define the concept BulkExecutor

BulkExecutors can schedule the invocation lambdas in bulk via the bulk_add(size_t, function) member function
* the first parameter actually needs to be std::sizeN in general

BulkExecutors have a execution category typedef
* the execution_category describes the ordering of lambda invocations within a single bulk_add call

How to specify the domain of lambda execution?

* BulkExecutors could have a domain_type typedef to describe the indices of execution agents created
* alternatively we could receive a std::bounds type i.e., insist that execution agents are 0-indexed
* it might be too heavy-handed to require BulkExecutors authors to include this typedef
  really all you want is for them to define the execution category typedef & bulk_add() member function
  instead, maybe we could find a way to report the nesting depth of the execution category
  then, bulk_add could simply recieve a Tuple-like object whose size is the nesting depth:

  struct my_minimal_bulk_executor
  {
    typedef nested_execution_tag<parallel_execution_tag, concurrent_execution_tag> execution_category;

    // requirement: tuple_size_v<Tuple> == nesting depth of execution_category
    template<typename Tuple, typename Function>
    void bulk_add(Tuple bounds, Function f);
  }

function receives the argument domain_type::index_type
  * the function needs to receive a tuple of indices for nested execution

* alternatively the function would receive an execution group and BulkExecutors would need an execution_group_type typedef
* this would make BulkExecutors responsible for reasoning about the types of execution agents
* it wouldn't leave much for execution policies to do

async works by comparing the execution category of the BulkExecutor to the category of the execution policy
* if there is a match, we defer work creation to the BulkExecutor
* if there is not an exact match, we have to do some sort of lowering

we need to add the execution category nested_execution_tag<outer_tag, inner_tag>
* perhaps std::par_vec could be the typedef nested_execution_tag<parallel_execution_tag, vector_execution_tag>
* should nested_execution_tag be two-argument or variadic?

what is the execution category of groups? are they nested?

we could create executor adaptors that e.g. create parallel executors from lower-level concurrent executors
we could create executor adaptors that e.g. create parallel executors by flattening nested executors
we could create executor adaptors that e.g. create parallel executors directly from sequential executors
we could create executor adaptors that e.g. create parallel executors from simple non-bulk Executors
this is cool because we could have user-programmable decomposition this way

basically all the logic of parallelism goes into the user-programmable BulkExecutor

=====

This design induces the following analogies:

Allocators                    :: Executors
Iterators                     :: Execution Agents
Iterator Traversal Categories :: Execution Categories
Containers                    :: Execution Policies
Algorithms                    :: Algorithms

