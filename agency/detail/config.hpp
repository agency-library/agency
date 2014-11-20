#pragma once

#if __cplusplus > 201103L
#define __AGENCY_CONSTEXPR_AFTER_CXX11 constexpr
#else
#define __AGENCY_CONSTEXPR_AFTER_CXX11
#endif

#ifdef __CUDACC__
#  define __AGENCY_ANNOTATION __host__ __device__
#else
#  define __AGENCY_ANNOTATION __AGENCY_CONSTEXPR_AFTER_CXX11
#endif // __AGENCY_ANNOTATION

