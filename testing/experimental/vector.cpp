#include <iostream>
#include <cassert>

void test_fill_construct()
{
  using namespace agency::experimental;

  {
    // test fill construct empty vector
    std::vector<int> v(0, 13);

    assert(v.empty());
  }
}

int main()
{
  test_fill_construct();

  std::cout << "OK" << std::endl;

  return 0;
}

