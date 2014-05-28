#pragma once

#include <future>
#include <utility>
#include <processor>

class async_launcher
{
  public:
    template<class Function, class... Args>
    std::future<void> async(Function&& f, Args&&... args) const
    {
      return std::async(std::launch::async, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    template<class Function, class... Args>
    void sync(Function&& f, Args&&... args) const
    {
      std::forward<Function>(f)(std::forward<Args>(args)...);
    }
};

template<class ProcessorID = std::processor_id>
class processor_launcher
{
  public:
    processor_launcher(ProcessorID proc)
      : proc_(proc)
    {}

    template<class Function, class... Args>
    std::future<void> async(Function&& f, Args&&... args) const
    {
      return std::async(proc_, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    template<class Function, class... Args>
    void sync(Function&& f, Args&&... args) const
    {
      std::sync(proc_, std::forward<Function>(f), std::forward<Args>(args)...);
    }

  private:
    ProcessorID proc_;
};

