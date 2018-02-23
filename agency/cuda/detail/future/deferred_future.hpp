#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/cuda/detail/boxed_value.hpp>
#include <agency/detail/unit.hpp>
#include <agency/detail/factory.hpp>
#include <agency/future.hpp>
#include <agency/cuda/detail/future/async_future.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/unique_function.hpp>
#include <agency/memory/allocator/detail/malloc_allocator.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
struct is_trivial_and_empty :
  agency::detail::conjunction<
    // XXX stdlibc++ doesn't have this yet???
    //std::is_trivially_default_constructible<T>,
    std::is_trivially_destructible<T>,
    std::is_empty<T>
>
{};


template<class T>
struct deferred_state_requires_storage
  : std::integral_constant<
      bool,
      !std::is_void<T>::value && !is_trivial_and_empty<T>::value
    >
{};


template<class T>
struct deferred_function_result : std::conditional<
  deferred_state_requires_storage<T>::value,
  T,
  agency::detail::unit
>
{};


template<class>
class deferred_function;

template<class Result, class... Args>
class deferred_function<Result(Args...)>
{
  private:
    template<class> friend class deferred_function;

    using result_type = typename deferred_function_result<Result>::type;
    using allocator_type = agency::detail::malloc_allocator<void>;

    agency::detail::unique_function<result_type(Args...)> function_;

    template<class Function>
    struct invoke_and_return_unit
    {
      mutable Function function_;

      __AGENCY_ANNOTATION
      agency::detail::unit operator()(Args&&... args) const
      {
        agency::detail::invoke(function_, std::forward<Args>(args)...);
        return agency::detail::unit{};
      }
    };

    template<class Function>
    __AGENCY_ANNOTATION
    invoke_and_return_unit<Function> make_invoke_and_return_unit(Function&& f)
    {
      return invoke_and_return_unit<Function>{std::forward<Function>(f)};
    }

  public:
    deferred_function() = default;

    deferred_function(deferred_function&& other) = default;

    deferred_function& operator=(deferred_function&& other) = default;

    template<class Function>
    __AGENCY_ANNOTATION
    deferred_function(Function&& f,
                      typename std::enable_if<
                        !deferred_state_requires_storage<
                          agency::detail::result_of_t<Function(Args...)>
                        >::value
                      >::type* = 0)
      : function_(std::allocator_arg, allocator_type(), make_invoke_and_return_unit(std::forward<Function>(f)))
    {}

    template<class Function>
    __AGENCY_ANNOTATION
    deferred_function(Function&& f,
                      typename std::enable_if<
                        deferred_state_requires_storage<
                          agency::detail::result_of_t<Function(Args...)>
                        >::value
                      >::type* = 0)
      : function_(std::allocator_arg, allocator_type(), std::forward<Function>(f))
    {}

    // allow moves from other deferred_function if their result_type is the same as ours
    template<class OtherResult,
             class = typename std::enable_if<
               std::is_same<
                 result_type, typename deferred_function<OtherResult(Args...)>::result_type
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_function(deferred_function<OtherResult(Args...)>&& other)
      : function_(std::move(other.function_))
    {
    }

    __AGENCY_ANNOTATION
    result_type operator()(Args... args) const
    {
      return function_(args...);
    }

    __AGENCY_ANNOTATION
    operator bool () const
    {
      return function_;
    }
};


struct ready_made_t {};
constexpr ready_made_t ready_made{};


template<class T, bool = deferred_state_requires_storage<T>::value>
class deferred_result : detail::boxed_value<agency::experimental::optional<T>>
{
  public:
    using super_t = detail::boxed_value<agency::experimental::optional<T>>;
    using super_t::super_t;
    using super_t::operator=;

    using value_type = T;

    __AGENCY_ANNOTATION
    deferred_result(deferred_result&& other)
      : super_t(std::move(static_cast<super_t&>(other)))
    {
      // empty other
      other.value() = agency::experimental::nullopt;
    }

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<
                 T, Args&&...
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_result(ready_made_t, Args&&... args)
      : super_t(std::forward<Args>(args)...)
    {}

    __AGENCY_ANNOTATION
    deferred_result& operator=(deferred_result&& other)
    {
      super_t::operator=(std::move(other));

      // empty other
      other.value() = agency::experimental::nullopt;

      return *this;
    }

    __AGENCY_ANNOTATION
    bool is_ready() const
    {
      return static_cast<bool>(this->value());
    }

    __AGENCY_ANNOTATION
    T& ref()
    {
      return this->value().value();
    }

  private:
    struct invalidate_at_scope_exit
    {
      deferred_result& self;

      __AGENCY_ANNOTATION
      ~invalidate_at_scope_exit()
      {
        self.value() = agency::experimental::nullopt;
      }
    };

  public:
    // calls f on the object contained within this deferred_result
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(agency::detail::invoke(std::forward<Function>(f), this->ref()))
    {
      return agency::detail::invoke(std::forward<Function>(f), ref());
    }

    // calls f on the object contained within this deferred_result
    // and invalidates this deferred_result
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap_and_invalidate(Function&& f) ->
      decltype(this->fmap(std::forward<Function>(f)))
    {
      invalidate_at_scope_exit invalidator{*this};
      return fmap(std::forward<Function>(f));
    }
};


template<class T>
class deferred_result<T,false>
  : agency::experimental::optional<
      typename std::conditional<
        std::is_void<T>::value,
        agency::detail::unit,
        T
      >::type
    >
{
  private:
    template<class,bool> friend class deferred_result;

  public:
    using value_type = typename std::conditional<
      std::is_void<T>::value,
      agency::detail::unit,
      T
    >::type;

    using super_t = agency::experimental::optional<value_type>;

    using super_t::super_t;
    using super_t::operator=;

    __AGENCY_ANNOTATION
    deferred_result()
      : super_t{}
    {}

    __AGENCY_ANNOTATION
    deferred_result(deferred_result&& other)
      : super_t(std::move(other))
    {
      // empty other
      other = agency::experimental::nullopt;
    }

    template<class U,
             class = typename std::enable_if<
               !deferred_state_requires_storage<U>::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_result(deferred_result<U>&& other)
      : deferred_result()
    {
      if(other.is_ready())
      {
        // ready ourself
        *this = value_type{};
      }

      // empty other
      other = agency::experimental::nullopt;
    }

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<
                 T, Args&&...
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_result(ready_made_t, Args&&... args)
      : super_t(value_type{std::forward<Args>(args)...}) // note we explicitly construct a value_type here to ensure we get the correct super_t constructor
    {
    }

    __AGENCY_ANNOTATION
    deferred_result& operator=(deferred_result&& other)
    {
      super_t::operator=(std::move(other));

      // empty other
      other = agency::experimental::nullopt;

      return *this;
    }

    __AGENCY_ANNOTATION
    bool is_ready() const
    {
      return static_cast<bool>(*this);
    }

  private:
    using reference = typename agency::detail::lazy_conditional<
      std::is_void<T>::value,
      agency::detail::identity<void>,
      std::add_lvalue_reference<T>
    >::type;

    template<class U>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      !std::is_void<U>::value,
      reference
    >::type
      ref_impl() const
    {
      deferred_result& self = const_cast<deferred_result&>(*this);
      return *self;
    }

    template<class U>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      std::is_void<U>::value,
      void
    >::type
      ref_impl() const
    {
    }

  public:
    __AGENCY_ANNOTATION
    reference ref()
    {
      return ref_impl<T>();
    }

  private:
    struct invalidate_at_scope_exit
    {
      deferred_result& self;

      __AGENCY_ANNOTATION
      ~invalidate_at_scope_exit()
      {
        self = agency::experimental::nullopt;
      }
    };

    template<class U, class Function,
             class = typename std::enable_if<
               std::is_void<U>::value
             >::type>
    __AGENCY_ANNOTATION
    auto fmap_impl(Function&& f) ->
      decltype(agency::detail::invoke(std::forward<Function>(f)))
    {
      return agency::detail::invoke(std::forward<Function>(f));
    }

    template<class U, class Function,
             class = typename std::enable_if<
               !std::is_void<U>::value
             >::type>
    __AGENCY_ANNOTATION
    auto fmap_impl(Function&& f) ->
      decltype(agency::detail::invoke(std::forward<Function>(f), this->ref()))
    {
      return agency::detail::invoke(std::forward<Function>(f), this->ref());
    }


  public:
    // calls f
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(this->template fmap_impl<T>(std::forward<Function>(f)))
    {
      return fmap_impl<T>(std::forward<Function>(f));
    }

    // calls f and invalidates this deferred_result
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap_and_invalidate(Function&& f) ->
      decltype(this->fmap(std::forward<Function>(f)))
    {
      invalidate_at_scope_exit invalidator{*this};
      return fmap(std::forward<Function>(f));
    }
};


template<class T>
class deferred_state
{
  private:
    template<class> friend class deferred_state;

    deferred_result<T> result_;

    using function_type = deferred_function<T()>;
    function_type function_;

    template<class U>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      !deferred_state_requires_storage<U>::value,
      typename deferred_result<T>::value_type
    >::type
      invoke_function()
    {
      function_();
      return typename deferred_result<T>::value_type{};
    }

    template<class U>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      deferred_state_requires_storage<U>::value,
      typename deferred_result<T>::value_type
    >::type
      invoke_function()
    {
      return function_();
    }

  public:
    deferred_state() = default;

    deferred_state(deferred_state&&) = default;

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<
                 deferred_result<T>, deferred_result<U>&&
               >::value &&
               std::is_constructible<
                 function_type, typename deferred_state<U>::function_type&&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_state(deferred_state<U>&& other)
      : result_(std::move(other.result_)),
        function_(std::move(other.function_))
    {
    }

    template<class Function,
             class = typename std::enable_if<
               std::is_constructible<
                 function_type, Function&&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_state(Function&& f)
      : function_(std::forward<Function>(f))
    {}

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args&&...>::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_state(ready_made_t, Args&&... args)
      : function_{},
        result_{ready_made, std::forward<Args>(args)...}
    {
    }

    deferred_state& operator=(deferred_state&& other) = default;

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return function_ || result_.is_ready();
    }

    __AGENCY_ANNOTATION
    bool is_ready() const
    {
      return valid() && result_.is_ready();
    }

    __AGENCY_ANNOTATION
    void wait()
    {
      if(!is_ready())
      {
        result_ = invoke_function<T>();

        // empty the function
        function_ = function_type();
      }
    }

  private:
    struct move_functor
    {
      template<class U>
      __AGENCY_ANNOTATION
      U operator()(U& arg) const
      {
        return std::move(arg); 
      }

      __AGENCY_ANNOTATION
      void operator()() const {}
    };

  public:
    // waits for the state to become ready and calls f on the object contained within the state, if any
    // invalidates the state
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(result_.fmap_and_invalidate(std::forward<Function>(f)))
    {
      wait();
      return result_.fmap_and_invalidate(std::forward<Function>(f));
    }

    // waits for the state to become ready and calls f on the object contained within the state, if any
    // leaves the state valid
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap_and_leave_valid(Function&& f) ->
      decltype(result_.fmap(std::forward<Function>(f)))
    {
      wait();
      return result_.fmap(std::forward<Function>(f));
    }

    // waits for the state to become ready and move-constructs the object contained within the state, if any
    // invalidates the state
    __AGENCY_ANNOTATION
    T get()
    {
      return fmap(move_functor());
    }

  private:
    struct get_ref_functor
    {
      template<class U>
      __AGENCY_ANNOTATION
      U& operator()(U& arg) const
      {
        return arg;
      }

      __AGENCY_ANNOTATION
      void operator()() const
      {
      }
    };

  public:
    // waits for the state to become ready and returns a reference to the object contained within the state, if any
    // leaves the state valid
    __AGENCY_ANNOTATION
    auto get_ref() ->
      decltype(this->fmap_and_leave_valid(get_ref_functor()))
    {
      return fmap_and_leave_valid(get_ref_functor());
    }
};


// XXX why is this functor external to deferred_future?
template<class Function, class DeferredFuture>
struct deferred_continuation
{
  mutable Function f_;
  mutable DeferredFuture predecessor_;

  __AGENCY_ANNOTATION
  auto operator()() const ->
    decltype(predecessor_.fmap(f_))
  {
    return predecessor_.fmap(f_);
  }
};


} // end detail


template<class T>
class deferred_future
{
  public:
    deferred_future() = default;

    deferred_future(deferred_future&&) = default;

    deferred_future& operator=(deferred_future&& other) = default;

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<
                 detail::deferred_state<T>,detail::deferred_state<U>&&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_future(deferred_future<U>&& other)
      : state_(std::move(other.state_))
    {
    }

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return state_.valid();
    }

    __AGENCY_ANNOTATION
    bool is_ready() const
    {
      return state_.is_ready();
    }

    __AGENCY_ANNOTATION
    void wait()
    {
      state_.wait();
    }

    __AGENCY_ANNOTATION
    T get()
    {
      return state_.get();
    }

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __AGENCY_ANNOTATION
    static deferred_future make_ready(Args&&... args)
    {
      return deferred_future(detail::ready_made, std::forward<Args>(args)...);
    }

    template<class Function>
    __AGENCY_ANNOTATION
    deferred_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        deferred_future
      >
    >
      then(Function&& f)
    {
      auto continuation = detail::deferred_continuation<typename std::decay<Function>::type, deferred_future>{std::forward<Function>(f), std::move(*this)};
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type, deferred_future>;
      return deferred_future<result_type>(std::move(continuation));
    }

  private:
    detail::deferred_state<T> state_;

    template<class>
    friend class deferred_future;

    template<class,class>
    friend struct agency::cuda::detail::deferred_continuation;

  // XXX leave this public so uber_future can use it
  //     make it private once uber_future's functionality has been fully integrated into the codebase
  public:
    __AGENCY_ANNOTATION
    auto get_ref() ->
      decltype(state_.get_ref())
    {
      return state_.get_ref();
    }

  private:
    template<class Function,
             class = typename std::enable_if<
               std::is_constructible<
                 detail::deferred_state<T>,
                 Function&&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_future(Function&& f)
      : state_(std::forward<Function>(f))
    {}

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args&&...>::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_future(detail::ready_made_t, Args&&... args)
      : state_(detail::ready_made, std::forward<Args>(args)...)
    {
    }

    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(state_.fmap(std::forward<Function>(f)))
    {
      return state_.fmap(std::forward<Function>(f));
    }

    template<class Function>
    struct then_and_leave_valid_functor
    {
      mutable Function f_;
      deferred_future& predecessor_;
    
      __AGENCY_ANNOTATION
      auto operator()() const ->
        decltype(predecessor_.state_.fmap_and_leave_valid(f_))
      {
        return predecessor_.state_.fmap_and_leave_valid(f_);
      }
    };

  // XXX leave these public so uber_future can use them
  //     make them private once uber_future's functionality has been fully integrated into the codebase
  public:
    template<class Function>
    __AGENCY_ANNOTATION
    deferred_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        deferred_future
      >
    >
      then_and_leave_valid(Function&& f)
    {
      auto continuation = then_and_leave_valid_functor<typename std::decay<Function>::type>{std::forward<Function>(f), *this};
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type, deferred_future>;
      return deferred_future<result_type>(std::move(continuation));
    }

    template<class Function, class Shape, class IndexFunction, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    deferred_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_and_leave_valid(Function, Shape, IndexFunction, ResultFactory, OuterFactory, InnerFactory, device_id)
    {
      printf("deferred_future::bulk_then_and_leave_valid(): Unimplemented.\n");

      return deferred_future<agency::detail::result_of_t<ResultFactory()>>();
    }
};

} // end cuda
} // end agency

