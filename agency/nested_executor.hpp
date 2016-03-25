#pragma once

#include <utility>
#include <agency/executor_array.hpp>

namespace agency
{


// XXX we should rename this to something like scoped_executor
template<class Executor1, class Executor2>
class nested_executor : public executor_array<Executor2,Executor1>
{
  private:
    using super_t = executor_array<Executor2,Executor1>;

  public:
    using outer_executor_type = Executor1;
    using inner_executor_type = Executor2;

    nested_executor(const outer_executor_type&,
                    const inner_executor_type& inner_ex)
      : super_t(1, inner_ex)
    {}

    nested_executor() :
      nested_executor(outer_executor_type(), inner_executor_type())
    {}
};


} // end agency

