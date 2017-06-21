/// \example saxpy.cpp
/// \brief Demonstrates how to use `bulk_invoke` to perform a parallel SAXPY.
///

#include <agency/agency.hpp>
#include <vector>
#include <cassert>
#include <iostream>

int main()
{
  // set up some inputs
  size_t n = 16 << 20;
  std::vector<float> x(n, 1), y(n, 2);
  float a = 13.;

  // use par to execute SAXPY in parallel, and collect the results
  auto results = agency::bulk_invoke(agency::par(n), [&](agency::parallel_agent& self)
  {
    int i = self.index();
    return a * x[i] + y[i];
  });

  // check the result
  std::vector<float> ref(n, a * 1.f + 2.f);
  assert(results == ref);

  std::cout << "OK" << std::endl;

  return 0;
}

