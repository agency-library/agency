#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_array.hpp>

namespace agency
{


template<class Executor1, class Executor2>
class scoped_executor : public executor_array<Executor2,Executor1>
{
  private:
    using super_t = executor_array<Executor2,Executor1>;

  public:
    using outer_executor_type = Executor1;
    using inner_executor_type = Executor2;

    scoped_executor(const outer_executor_type&,
                    const inner_executor_type& inner_ex)
      : super_t(1, inner_ex)
    {}

    scoped_executor() :
      scoped_executor(outer_executor_type(), inner_executor_type())
    {}
};


} // end agency

