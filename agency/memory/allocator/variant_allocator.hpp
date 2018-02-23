#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>

namespace agency
{


template<class Alloc, class... Allocs>
class variant_allocator
{
  private:
    using variant_type = agency::experimental::variant<Alloc, Allocs...>;

  public:
    using value_type = typename std::allocator_traits<Alloc>::value_type;

    template<class T>
    struct rebind
    {
      using other = variant_allocator<
        typename std::allocator_traits<Alloc>::template rebind_alloc<T>,
        typename std::allocator_traits<Allocs>::template rebind_alloc<T>...
      >;
    };

    variant_allocator() = default;

    variant_allocator(const variant_allocator&) = default;

  public:
    // this constructor converts from another allocator, when possible
    template<class OtherAlloc,
             __AGENCY_REQUIRES(
               std::is_constructible<variant_type, const OtherAlloc&>::value
             )>
    __AGENCY_ANNOTATION
    variant_allocator(const OtherAlloc& alloc)
      : variant_(alloc)
    {}

  private:
    template<class, class...>
    friend class variant_allocator;

    template<class OtherAlloc, class... OtherAllocs>
    struct converting_constructor_visitor
    {
      template<class A>
      __AGENCY_ANNOTATION
      variant_type operator()(const A& alloc) const
      {
        // lookup the index of A in <OtherAlloc, OtherAllocs...>
        constexpr std::size_t index = agency::experimental::detail::variant_detail::find_type<A, OtherAlloc, OtherAllocs...>::value;

        // lookup the type of the corresponding allocator in our variant
        using target_allocator_type = agency::experimental::variant_alternative_t<index, variant_type>;

        return static_cast<target_allocator_type>(alloc);
      }
    };

  public:
    // this constructor converts from another variant_allocator
    template<class OtherAlloc, class... OtherAllocs,
             __AGENCY_REQUIRES(sizeof...(Allocs) == sizeof...(OtherAllocs)),
             __AGENCY_REQUIRES(
               detail::conjunction<
                 std::is_constructible<Alloc,const OtherAlloc&>,
                 std::is_constructible<Allocs,const OtherAllocs&>...
               >::value
             )>
    __AGENCY_ANNOTATION
    variant_allocator(const variant_allocator<OtherAlloc,OtherAllocs...>& other)
      : variant_(experimental::visit(converting_constructor_visitor<OtherAlloc,OtherAllocs...>{}, other.variant_))
    {}

  private:
    struct allocate_visitor
    {
      std::size_t n;

      template<class A>
      __AGENCY_ANNOTATION
      value_type* operator()(A& alloc) const
      {
        return detail::allocator_traits<A>::allocate(alloc, n);
      }
    };

  public:
    __AGENCY_ANNOTATION
    value_type* allocate(std::size_t n)
    {
      return experimental::visit(allocate_visitor{n}, variant_);
    }

  private:
    struct deallocate_visitor
    {
      value_type* ptr;
      std::size_t n;

      template<class A>
      __AGENCY_ANNOTATION
      void operator()(A& alloc) const
      {
        detail::allocator_traits<A>::deallocate(alloc, ptr, n);
      }
    };

  public:
    __AGENCY_ANNOTATION
    void deallocate(value_type* ptr, std::size_t n)
    {
      return experimental::visit(deallocate_visitor{ptr,n}, variant_);
    }

#if __cpp_generic_lambdas
    template<class T, class... Args>
    __AGENCY_ANNOTATION
    void construct(T* ptr, Args&&... args)
    {
      return experimental::visit([&](auto& alloc)
      {
        using allocator_type = decltype(alloc);
        return detail::allocator_traits<allocator_type>::construct(alloc, ptr, std::forward<Args>(args)...);
      },
      variant_);
    }
#endif

  private:
    template<class T>
    struct destroy_visitor
    {
      T* ptr;

      template<class A>
      __AGENCY_ANNOTATION
      void operator()(A& alloc) const
      {
        detail::allocator_traits<A>::destroy(alloc, ptr);
      }
    };

  public:
    template<class T>
    __AGENCY_ANNOTATION
    void destroy(T* ptr)
    {
      experimental::visit(destroy_visitor<T>{ptr}, variant_);
    }

  private:
    struct max_size_visitor
    {
      template<class A>
      __AGENCY_ANNOTATION
      std::size_t operator()(const A& alloc) const
      {
        return detail::allocator_traits<A>::max_size(alloc);
      }
    };

  public:
    __AGENCY_ANNOTATION
    std::size_t max_size() const
    {
      return experimental::visit(max_size_visitor{}, variant_);
    }

    __AGENCY_ANNOTATION
    bool operator==(const variant_allocator& other) const
    {
      return variant_ == other.variant_;
    }

    __AGENCY_ANNOTATION
    bool operator!=(const variant_allocator& other) const
    {
      return variant_ != other.variant_;
    }

  private:
    variant_type variant_;
};


} // end agency

