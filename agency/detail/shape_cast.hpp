#pragma once

#include <tuple>
#include <utility>
#include <type_traits>
#include <agency/coordinate.hpp>
#include <agency/detail/tuple_utility.hpp>
#include <agency/detail/point_size.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{
namespace detail
{
namespace shape_cast_detail
{


template<class T>
struct make
{
  template<class... Args>
  __AGENCY_ANNOTATION
  T operator()(Args&&... args) const
  {
    return T{std::forward<Args>(args)...};
  }
};


} // end shape_cast_detail


// reduces the dimensionality of x by eliding the last dimension
// and multiplying the second-to-last dimension by the last 
template<class Point>
__AGENCY_ANNOTATION
rebind_point_size_t<
  Point,
  point_size<Point>::value - 1
> project_shape(const Point& x)
{
  using result_type = rebind_point_size_t<Point,std::tuple_size<Point>::value-1>;

  auto last = __tu::tuple_last(x);

  // XXX WAR nvcc 7's issue with tuple_drop_invoke
  //auto result = __tu::tuple_drop_invoke<1>(x, shape_cast_detail::make<result_type>());
  auto result = __tu::tuple_take_invoke<std::tuple_size<Point>::value - 1>(x, shape_cast_detail::make<result_type>());

  __tu::tuple_last(result) *= last;

  return result;
}


// increases the dimensionality of x
// by appending a dimension (and setting it to 1)
template<class Point>
__AGENCY_ANNOTATION
rebind_point_size_t<
  Point,
  point_size<Point>::value + 1
> lift_shape(const Point& x)
{
  // x could be a scalar, so create an intermediate which is at least a 1-element tuple
  using intermediate_type = rebind_point_size_t<Point,point_size<Point>::value>;
  intermediate_type intermediate{x};

  using result_type = rebind_point_size_t<Point,point_size<Point>::value + 1>;

  return __tu::tuple_append_invoke(intermediate, 1, shape_cast_detail::make<result_type>());
}


// __shape_cast is recursive and has various overloads
// declare them here before their definitions

// Scalar -> Scalar (base case)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value == 1),
  ToShape
>::type
  shape_cast(const FromShape& x);


// recursive case for casting two shapes of equal size (recursive case)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value > 1),
  ToShape
>::type
  shape_cast(const FromShape& x);


// downcast (recursive)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value < point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x);


// upcast (recursive)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value > point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x);


// definitions of __shape_cast follow


// terminal case for casting shapes of size 1
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value == 1),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  // x might not be a tuple, but instead a scalar type
  // to ensure we can get the 0th value from x in a uniform way, lift it first
  return static_cast<ToShape>(detail::get<0>(lift_shape(x)));
}

struct shape_cast_functor
{
  template<class ToShape, class FromShape>
  __AGENCY_ANNOTATION
  auto operator()(const ToShape&, const FromShape& x)
    -> decltype(
         shape_cast<ToShape>(x)
       )
  {
    return shape_cast<ToShape>(x);
  }
};


// recursive case for casting to a shape of equal size
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value > 1),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  return __tu::tuple_map_with_make(shape_cast_functor{}, shape_cast_detail::make<ToShape>{}, ToShape{}, x);
}


// recursive case for casting to a lower dimensional shape
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value < point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  return shape_cast<ToShape>(project_shape(x));
}


// recursive case for casting to a higher dimensional shape
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value > point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  return shape_cast<ToShape>(lift_shape(x));
}


} // end detail
} // end agency

