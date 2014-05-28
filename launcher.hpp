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

template<class Function, class... Args>
std::future<void> async(const async_launcher& launcher, Function&& f, Args&&... args)
{
  return launcher.async(std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
void sync(const async_launcher& launcher, Function&& f, Args&&... args)
{
  return launcher.sync(std::forward<Function>(f), std::forward<Args>(args)...);
}

