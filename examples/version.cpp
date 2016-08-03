#include <agency/agency.hpp>
#include <iostream>

int main()
{
  int major = AGENCY_MAJOR_VERSION;
  int minor = AGENCY_MINOR_VERSION;

  std::cout << "Agency v" << major << "." << minor << std::endl;

  return 0;
}

