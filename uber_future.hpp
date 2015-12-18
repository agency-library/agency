#pragma once

#include <agency/detail/config.hpp>
#include "deferred_future.hpp"
#include <agency/cuda/future.hpp>
#include <agency/detail/variant.hpp>

template<class T>
class uber_future
{
  private:
    using variant_type = agency::detail::variant<deferred_future<T>, agency::cuda::future<T>>;

  public:
    __AGENCY_ANNOTATION
    uber_future() = default;

    __AGENCY_ANNOTATION
    uber_future(uber_future&&) = default;

    template<class Future,
             class = typename std::enable_if<
               std::is_constructible<variant_type,Future&&>::value
             >::type>
    __AGENCY_ANNOTATION
    uber_future(Future&& other)
      : variant_(std::forward<Future>(other))
    {}

    __AGENCY_ANNOTATION
    uber_future& operator=(uber_future&& other) = default;

  private:
    struct valid_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      bool operator()(const Future& f) const
      {
        return f.valid();
      }
    };

  public:
    __AGENCY_ANNOTATION
    bool valid() const
    {
      return agency::detail::visit(valid_visitor(), variant_);
    }

  private:
    struct wait_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      void operator()(Future& f) const
      {
        f.wait();
      }
    };

  public:
    __AGENCY_ANNOTATION
    void wait()
    {
      return agency::detail::visit(wait_visitor(), variant_);
    }

  private:
    struct get_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      T operator()(Future& f) const
      {
        return f.get();
      }
    };

  public:
    __AGENCY_ANNOTATION
    T get()
    {
      return agency::detail::visit(get_visitor(), variant_);
    }

    template<class... Args,
             class = typename std::enable_if<
               agency::cuda::detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __AGENCY_ANNOTATION
    static uber_future make_ready(Args&&... args)
    {
      return deferred_future<T>::make_ready(std::forward<Args>(args)...);
    }

  private:
    template<class Function>
    struct then_visitor
    {
      mutable Function f;

      template<class Future>
      __AGENCY_ANNOTATION
      uber_future<
        agency::detail::result_of_continuation_t<
          Function, 
          Future
        >
      >
        operator()(Future& fut) const
      {
        return fut.then(std::move(f));
      }
    };

  public:
    template<class Function>
    __AGENCY_ANNOTATION
    uber_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        uber_future
      >
    >
      then(Function&& f)
    {
      auto visitor = then_visitor<typename std::decay<Function>::type>{std::forward<Function>(f)};
      return agency::detail::visit(visitor, variant_);
    }

  private:
    variant_type variant_;
};

