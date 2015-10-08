#include <vector>
#include <cassert>
#include <iostream>
#include <agency/execution_policy.hpp>

void saxpy(size_t n, float a, const float* x, const float* y, float* z)
{
  using namespace agency;

  bulk_invoke(par(n), [=](parallel_agent &self)
  {
    int i = self.index();
    z[i] = a * x[i] + y[i];
  });
}

int main()
{
  size_t n = 1 << 30;
  std::vector<float> x(n, 1), y(n, 2), z(n);
  float a = 13.;

  saxpy(n, a, x.data(), y.data(), z.data());

  std::vector<float> ref(n, a * 1.f + 2.f);
  assert(ref == z);

  std::cout << "OK" << std::endl;

  return 0;
}

