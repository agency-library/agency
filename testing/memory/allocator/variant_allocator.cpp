#include <agency/memory/allocator/variant_allocator.hpp>
#include <vector>
#include <deque>
#include <forward_list>
#include <memory>
#include <algorithm>
#include <iostream>
#include <cassert>

int allocate_counter = 0;
int deallocate_counter = 0;

template<class T>
struct my_allocator
{
  using value_type = T;

  my_allocator() = default;

  template<class U>
  my_allocator(const my_allocator<U>&) {}

  value_type* allocate(std::size_t n)
  {
    ++allocate_counter;
    return static_cast<value_type*>(std::malloc(sizeof(value_type) * n));
  }

  void deallocate(value_type* ptr, std::size_t n)
  {
    ++deallocate_counter;
    std::free(ptr);
  }
};

template<class T>
using allocator = agency::variant_allocator<my_allocator<T>, std::allocator<T>>;

int main()
{
  {
    allocate_counter = deallocate_counter = 0;

    allocator<int> alloc = my_allocator<int>();
    
    // test my_allocator with std::vector
    std::vector<int, allocator<int>> vec(10, 13, alloc);

    assert(std::all_of(vec.begin(), vec.end(), [](int x)
    {
      return x == 13;
    }));

    vec.clear();
    vec.shrink_to_fit();

    assert(allocate_counter == 1);
    assert(deallocate_counter == 1);
  }

  {
    allocate_counter = deallocate_counter = 0;

    allocator<int> alloc = std::allocator<int>();
    
    // test std::allocator with std::vector
    std::vector<int, allocator<int>> vec(10, 13, alloc);

    assert(std::all_of(vec.begin(), vec.end(), [](int x)
    {
      return x == 13;
    }));

    vec.clear();
    vec.shrink_to_fit();

    assert(allocate_counter == 0);
    assert(deallocate_counter == 0);
  }

  {
    allocate_counter = deallocate_counter = 0;

    allocator<int> alloc = my_allocator<int>();
    
    // test my_allocator with std::forward_list
    std::forward_list<int, allocator<int>> list(10, 13, alloc);

    assert(std::all_of(list.begin(), list.end(), [](int x)
    {
      return x == 13;
    }));

    list.clear();

    assert(allocate_counter == 10);
    assert(deallocate_counter == 10);
  }

  {
    allocate_counter = deallocate_counter = 0;

    allocator<int> alloc = std::allocator<int>();
    
    // test std::allocator with std::forward_list
    std::forward_list<int, allocator<int>> list(10, 13, alloc);

    assert(std::all_of(list.begin(), list.end(), [](int x)
    {
      return x == 13;
    }));

    list.clear();

    assert(allocate_counter == 0);
    assert(deallocate_counter == 0);
  }

  std::cout << "OK" << std::endl;
}

