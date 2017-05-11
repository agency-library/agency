#include <agency/memory/allocator/detail/any_allocator.hpp>
#include <agency/cuda.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

template<class T>
struct my_allocator
{
  using value_type = T;

  my_allocator() = default;

  template<class U>
  my_allocator(const my_allocator<U>&) {}

  T* allocate(std::size_t n)
  {
    return std::allocator<T>().allocate(n);
  }

  void deallocate(T* ptr, std::size_t n)
  {
    return std::allocator<T>().deallocate(ptr, n);
  }

  bool operator==(const my_allocator&) const
  {
    return true;
  }

  bool operator!=(const my_allocator&) const
  {
    return false;
  }
};

void test()
{
  {
    // test default constructed any_small_allocator
    
    agency::detail::any_small_allocator<int> alloc;

    std::vector<int, agency::detail::any_small_allocator<int>> vec(10, 13);
    assert(std::count(vec.begin(), vec.end(), 13) == 10);

    agency::detail::any_small_allocator<int> copy_of_alloc = alloc;
    assert(copy_of_alloc == alloc);
    assert(!(copy_of_alloc != alloc));
  }
}

template<class Allocator>
void test(const Allocator& other_alloc)
{
  {
    // test copying from an allocator with same value_type
    using value_type = typename Allocator::value_type;
    
    agency::detail::any_small_allocator<value_type> alloc = other_alloc;

    std::vector<value_type, agency::detail::any_small_allocator<value_type>> vec(10, 13);
    assert(std::count(vec.begin(), vec.end(), 13) == 10);

    agency::detail::any_small_allocator<value_type> copy_of_alloc = alloc;
    assert(copy_of_alloc == alloc);
    assert(!(copy_of_alloc != alloc));
  }

  {
    // test copying from an allocator with different value_type
    static_assert(!std::is_same<float, typename Allocator::value_type>::value, "For testing purposes, Allocator::value_type should not be float");
    
    agency::detail::any_small_allocator<float> alloc = other_alloc;

    std::vector<float, agency::detail::any_small_allocator<float>> vec(10, 13);
    assert(std::count(vec.begin(), vec.end(), 13) == 10);

    agency::detail::any_small_allocator<float> copy_of_alloc = alloc;
    assert(copy_of_alloc == alloc);
    assert(!(copy_of_alloc != alloc));
  }
}

int main()
{
  test(my_allocator<int>());
  test(agency::cuda::allocator<int>());

  std::cout << "OK" << std::endl;
}

