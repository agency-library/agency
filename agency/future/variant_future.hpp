#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/future/future_traits.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/future.hpp>

#include <type_traits>

namespace agency
{


template<class Future, class... Futures>
class variant_future
{
  private:
    using variant_type = agency::experimental::variant<Future, Futures...>;

  public:
    using result_type = future_result_t<Future>;

    template<class T>
    using rebind_value = variant_future<
      future_rebind_value_t<Future,T>,
      future_rebind_value_t<Futures,T>...
    >;

    static_assert(detail::conjunction<is_future<Future>, is_future<Futures>...>::value, "All of variant_future's template parmeter types must be Futures.");
    static_assert(detail::conjunction<std::is_same<result_type, future_result_t<Futures>>...>::value, "All Futures' result types must be the same.");

    variant_future() = default;

    variant_future(variant_future&&) = default;

    template<class OtherFuture,
             __AGENCY_REQUIRES(
               std::is_constructible<variant_type,OtherFuture&&>::value
            )>
    __AGENCY_ANNOTATION
    variant_future(OtherFuture&& other)
      : variant_(std::forward<OtherFuture>(other))
    {}

    variant_future& operator=(variant_future&& other) = default;

    // this is the overload of make_ready() for non-void result_type
    template<class T,
             __AGENCY_REQUIRES(
               std::is_constructible<result_type,T&&>::value
            )>
    static variant_future make_ready(T&& value)
    {
      // use the first Future type to create the ready state
      return future_traits<Future>::make_ready(std::forward<T>(value));
    }

    // this is the overload of make_ready() for void result_type
    template<bool deduced = true,
             __AGENCY_REQUIRES(
               deduced && std::is_void<result_type>::value
            )>
    static variant_future make_ready()
    {
      // use the first Future type to create the ready state
      return future_traits<Future>::make_ready();
    }

    /// XXX consider eliminating this member and instead deriving variant_future from variant
    __AGENCY_ANNOTATION
    size_t index() const
    {
      return variant_.index();
    }

    /// Returns this variant_future's underlying variant object and invalidates this variant_future.
    /// XXX consider eliminating this member and instead deriving variant_future from variant
    __AGENCY_ANNOTATION
    variant_type variant()
    {
      return std::move(variant_);
    }

  private:
    struct valid_visitor
    {
      template<class T>
      __AGENCY_ANNOTATION
      bool operator()(const T& f) const
      {
        return f.valid();
      }
    };

  public:
    __AGENCY_ANNOTATION
    bool valid() const
    {
      auto visitor = valid_visitor();
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    struct wait_visitor
    {
      __agency_exec_check_disable__
      template<class T>
      __AGENCY_ANNOTATION
      void operator()(T& f) const
      {
        f.wait();
      }
    };

  public:
    __AGENCY_ANNOTATION
    void wait()
    {
      auto visitor = wait_visitor();
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    struct get_visitor
    {
      __agency_exec_check_disable__
      template<class T>
      __AGENCY_ANNOTATION
      result_type operator()(T& f) const
      {
        return f.get();
      }
    };

  public:
    __AGENCY_ANNOTATION
    result_type get()
    {
      auto visitor = get_visitor();
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    template<class FunctionRef>
    struct then_visitor
    {
      FunctionRef f;

      template<class T>
      __AGENCY_ANNOTATION
      future_then_result_t<variant_future, detail::decay_t<FunctionRef>>
        operator()(T& future) const
      {
        // XXX should probably do this through a Future customization point
        return agency::future_traits<T>::then(future, std::forward<FunctionRef>(f));
      }
    };

  public:
    template<class Function>
    future_then_result_t<variant_future, Function>
      then(Function&& f)
    {
      auto visitor = then_visitor<Function&&>{std::forward<Function>(f)};
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    variant_type variant_;
};


namespace detail
{


template<class T>
struct is_variant_future : std::false_type {};


template<class Future, class... Futures>
struct is_variant_future<variant_future<Future,Futures...>> : std::true_type {};


} // end detail
} // end agency

