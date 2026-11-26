#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" __device__ void entry(double, double, double *, double *, void *, void *);
#ifdef PROFILE
extern "C" void __poseidon_profile_init_cuda();
#endif

__global__ void sample(double x, double dx, double *y, double *dy) {
  entry(x, dx, y, dy, nullptr, nullptr);
}

void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}

int main() {
#ifdef PROFILE
  __poseidon_profile_init_cuda();
#else
  std::puts("x,dx,value,jvp");
#endif
  double *out;
  check(cudaMalloc(&out, 2 * sizeof(double)));
  for (int i = 0; i < 32; ++i) {
#ifdef PROFILE
    double x = std::ldexp(1.125, i);
#else
    // Distinct from every training input and within its observed range.
    double x = std::ldexp(i == 31 ? 1.0625 : 1.375, i);
    double result[] = {NAN, NAN};
    check(cudaMemcpy(out, result, sizeof(result), cudaMemcpyHostToDevice));
#endif
    double dx = (i % 3 + 1) * 0.25;
    sample<<<1, 32>>>(x, dx, out, out + 1);
    check(cudaGetLastError());
#ifndef PROFILE
    check(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost));
    std::printf("%.17g,%.17g,%.17g,%.17g\n", x, dx, result[0], result[1]);
#endif
  }
  check(cudaDeviceSynchronize());
  check(cudaFree(out));
}
