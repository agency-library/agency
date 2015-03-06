#pragma once

#include <future>
#include <agency/execution_categories.hpp>
#include <functional>

namespace agency
{


class sequential_executor
{
  public:
    using execution_category = sequential_execution_tag;

    template<class Function, class T>
    void execute(Function f, size_t n, T shared_arg)
    {
      for(size_t i = 0; i < n; ++i)
      {
        f(i, shared_arg);
      }
    }

    template<class Function, class T>
    std::future<void> async_execute(Function f, size_t n, T shared_arg)
    {
      return std::async(std::launch::deferred, [=]
      {
        this->execute(f, n, shared_arg);
      });
    }
};


} // end agency

