# Parallel AES-128 with CUDA

**How fast is AES-128 on a GPU when every thread encrypts its own block, and when is it worth it?**

This project implements AES-128 (FIPS-197) for NVIDIA GPUs in CUDA C++. Each GPU thread processes one
16-byte block, so millions of blocks are encrypted in parallel, in ECB mode (encryption and decryption)
and in CTR mode (the counter mode of NIST SP 800-38A, where encryption and decryption are the same
operation).

| part | what it does |
| --- | --- |
| `src/aes_gpu.cuh` | the kernels: round keys in constant memory; each thread block builds its T-table and S-box in shared memory; one thread per block in a grid-stride loop; 16-byte vector loads and stores |
| `src/aes_cpu.h` | key expansion (including the round keys of the equivalent inverse cipher) and a byte-oriented reference AES used to check the GPU |
| `src/aes_cuda.cu` | stand-alone program: known-answer tests on the GPU, random data checked against the CPU reference, throughput |
| `tests/verify_gpu.py` | runs the same kernels through NVRTC (CuPy) and checks them against OpenSSL; needs no C++ host compiler |
| `tests/test_cpu_ref.cpp` | checks the CPU reference against FIPS-197 and SP 800-38A |

## Main results

Measured on two GPUs: an NVIDIA GeForce GTX 1650 (14 SMs, Windows) and a Tesla T4 (40 SMs, Google Colab,
Linux). Logs in `results/`.

1. **The kernels are correct.** They reproduce the FIPS-197 Appendix B and C.1 vectors and the
   SP 800-38A F.1.1/F.1.2 (ECB) and F.5.1/F.5.2 (CTR) vectors. On random data with a random key
   (256 MiB on the GTX 1650, 64 MiB on the T4), ECB encryption and CTR are byte-identical to OpenSSL,
   and ECB decryption inverts encryption. The CTR test starts the counter just below 2^64, so the carry
   into the upper half of the 128-bit counter is exercised.
2. **The stand-alone CUDA program builds and passes.** Compiled with `nvcc` on the T4, `src/aes_cuda.cu`
   passes the same known-answer tests and agrees with the CPU reference on all 4,194,304 blocks of
   64 MiB of random data, in ECB and CTR.
3. **Throughput.** Kernel time on data already in GPU memory, best of three runs after a warm-up
   (`tests/verify_gpu.py`):

   | GPU | ECB encrypt | ECB decrypt | CTR |
   | --- | ---: | ---: | ---: |
   | GTX 1650 | 12.9 GB/s | 12.9 GB/s | 14.6 GB/s |
   | Tesla T4 | 12.1 GB/s | 12.1 GB/s | 13.7 GB/s |

   The T4 has almost three times as many SMs, yet both cards reach about the same throughput; the
   kernels have not been profiled to find what limits them (the T4 runs at lower clocks within a 70 W
   power limit, which is one candidate). The stand-alone
   program times a single run without warm-up, so its figures vary more (on the T4 it once reported
   21.6 GB/s for ECB encryption); they are not used as the benchmark.
4. **For data that starts in host memory, one CPU core with AES-NI is faster.** Including the copies
   between host and GPU memory, the GTX 1650 reaches about 1.3 GB/s, while OpenSSL on one core of
   the test machine reaches 2.3 GB/s (ECB) and 3.4 GB/s (CTR) (`results/comparison.txt`). The GPU
   pays off when the data is produced or consumed on the GPU, or on CPUs without AES instructions.

## Folder map

```
parallel-aes-cuda/
  src/aes_gpu.cuh, src/aes_cpu.h, src/aes_cuda.cu
  tests/verify_gpu.py, tests/test_cpu_ref.cpp
  results/          GPU verification logs (GTX 1650, Tesla T4), stand-alone program on the T4,
                    CPU reference log, comparison with OpenSSL
```

## Building and running

With the CUDA toolkit (nvcc and a host compiler: GCC on Linux, Visual Studio on Windows), or on
Google Colab with a GPU runtime:

```bash
nvcc -O3 -std=c++17 -Isrc -o aes_cuda src/aes_cuda.cu
./aes_cuda 64            # known answers, 64 MiB of random data, throughput
```

Without a host compiler, the kernels can be compiled at run time with NVRTC:

```bash
pip install cupy-cuda12x cryptography     # cupy-cuda13x for CUDA 13
python tests/verify_gpu.py 64
```

The CPU reference alone:

```bash
g++ -std=c++17 -O2 tests/test_cpu_ref.cpp -o test_cpu_ref && ./test_cpu_ref
```

## Notes

- ECB mode is included because it is the simplest parallel mode and the one tested by FIPS-197; it
  should not be used to encrypt real data, since equal blocks give equal ciphertexts. CTR is the mode
  to use, normally with an authentication tag (as in GCM).
- The T-table implementation does table lookups that depend on the key and the data, so it is not
  protected against cache-timing side channels.

## Author

George David Tsitlauri, University of Thessaly.
