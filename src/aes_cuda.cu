// Parallel AES-128 on NVIDIA GPUs: self-test and benchmark.
//
//   nvcc -O3 -std=c++17 -o aes_cuda src/aes_cuda.cu
//   ./aes_cuda [MiB]          (default 64 MiB of random data)
//
// 1. Known-answer tests on the GPU: FIPS-197 Appendix B and C.1, NIST SP 800-38A
//    F.1.1 (ECB-AES128 encrypt), F.1.2 (decrypt) and F.5.1 (CTR-AES128).
// 2. Random data: the GPU result of ECB encryption, ECB decryption and CTR is
//    compared byte for byte with the CPU reference.
// 3. Throughput of each kernel (CUDA events, kernel time only) and of the CPU
//    reference on one core.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "aes_gpu.cuh"
#include "aes_cpu.h"

#define CHECK(call)                                                                  \
    do {                                                                             \
        cudaError_t e_ = (call);                                                     \
        if (e_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static const int THREADS = 256;

static void hex(const char *s, uint8_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) sscanf(s + 2 * i, "%2hhx", &out[i]);
}

static void set_key(const uint8_t key[16], uint32_t rk[44]) {
    uint32_t drk[44];
    aes_cpu::expand_key(key, rk);
    aes_cpu::expand_decrypt_key(rk, drk);
    CHECK(cudaMemcpyToSymbol(c_rk, rk, sizeof(uint32_t) * 44));
    CHECK(cudaMemcpyToSymbol(c_drk, drk, sizeof(uint32_t) * 44));
}

static int grid_for(size_t nblocks) {
    int dev, sms;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    size_t needed = (nblocks + THREADS - 1) / THREADS;
    size_t cap = (size_t)sms * 32;  // grid-stride loop beyond this
    return (int)(needed < cap ? (needed ? needed : 1) : cap);
}

enum Mode { ECB_ENC, ECB_DEC, CTR };

// Runs one kernel on host data; returns the kernel time in milliseconds.
static float run(Mode m, const uint8_t *in, uint8_t *out, size_t bytes, const uint8_t iv[16]) {
    size_t nblocks = bytes / 16;
    uint4 *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    int grid = grid_for(nblocks);
    CHECK(cudaEventRecord(a));
    if (m == ECB_ENC) aes128_ecb_encrypt<<<grid, THREADS>>>(d_in, d_out, nblocks);
    else if (m == ECB_DEC) aes128_ecb_decrypt<<<grid, THREADS>>>(d_in, d_out, nblocks);
    else {
        uint32_t w[4];
        for (int i = 0; i < 4; i++) w[i] = ((uint32_t)iv[4 * i] << 24) | ((uint32_t)iv[4 * i + 1] << 16) | ((uint32_t)iv[4 * i + 2] << 8) | iv[4 * i + 3];
        aes128_ctr_xcrypt<<<grid, THREADS>>>(d_in, d_out, nblocks, w[0], w[1], w[2], w[3]);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return ms;
}

static int failures = 0;

static void expect(const char *name, const uint8_t *got, const uint8_t *want, size_t n) {
    bool ok = memcmp(got, want, n) == 0;
    printf("  %-40s %s\n", name, ok ? "PASS" : "FAIL");
    failures += !ok;
}

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? (size_t)atoi(argv[1]) : 64;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs)\n\nKnown-answer tests\n", prop.name, prop.multiProcessorCount);

    uint32_t rk[44];
    uint8_t key[16], pt[64], ct[64], out[64], iv[16];

    hex("2b7e151628aed2a6abf7158809cf4f3c", key, 16);
    set_key(key, rk);
    hex("3243f6a8885a308d313198a2e0370734", pt, 16);
    hex("3925841d02dc09fbdc118597196a0b32", ct, 16);
    run(ECB_ENC, pt, out, 16, iv); expect("FIPS-197 Appendix B encrypt", out, ct, 16);
    run(ECB_DEC, ct, out, 16, iv); expect("FIPS-197 Appendix B decrypt", out, pt, 16);

    hex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710", pt, 64);
    hex("3ad77bb40d7a3660a89ecaf32466ef97f5d3d58503b9699de785895a96fdbaaf"
        "43b1cd7f598ece23881b00e3ed030688" "7b0c785e27e8ad3f8223207104725dd4", ct, 64);
    run(ECB_ENC, pt, out, 64, iv); expect("SP 800-38A F.1.1 ECB encrypt (4 blocks)", out, ct, 64);
    run(ECB_DEC, ct, out, 64, iv); expect("SP 800-38A F.1.2 ECB decrypt (4 blocks)", out, pt, 64);
    hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff", iv, 16);
    hex("874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff"
        "5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee", ct, 64);
    run(CTR, pt, out, 64, iv); expect("SP 800-38A F.5.1 CTR encrypt (4 blocks)", out, ct, 64);
    run(CTR, ct, out, 64, iv); expect("SP 800-38A F.5.2 CTR decrypt (4 blocks)", out, pt, 64);

    hex("000102030405060708090a0b0c0d0e0f", key, 16);
    set_key(key, rk);
    hex("00112233445566778899aabbccddeeff", pt, 16);
    hex("69c4e0d86a7b0430d8cdb78070b4c55a", ct, 16);
    run(ECB_ENC, pt, out, 16, iv); expect("FIPS-197 Appendix C.1 encrypt", out, ct, 16);
    run(ECB_DEC, ct, out, 16, iv); expect("FIPS-197 Appendix C.1 decrypt", out, pt, 16);

    // Random data against the CPU reference.
    size_t bytes = mib << 20;
    std::vector<uint8_t> data(bytes), gpu(bytes), back(bytes), ref(16);
    std::mt19937_64 rng(2026);
    for (size_t i = 0; i < bytes; i += 8) { uint64_t v = rng(); memcpy(&data[i], &v, 8); }
    for (int i = 0; i < 16; i++) { key[i] = (uint8_t)rng(); iv[i] = (uint8_t)rng(); }
    set_key(key, rk);
    printf("\nRandom data: %zu MiB, %zu blocks\n", mib, bytes / 16);

    float t_enc = run(ECB_ENC, data.data(), gpu.data(), bytes, iv);
    size_t bad = 0;
    auto c0 = std::chrono::high_resolution_clock::now();
    for (size_t b = 0; b < bytes; b += 16) {
        aes_cpu::encrypt_block_ref(rk, &data[b], ref.data());
        bad += memcmp(ref.data(), &gpu[b], 16) != 0;
    }
    double cpu_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - c0).count();
    printf("  ECB encrypt: %zu blocks differ from the CPU\n", bad);
    failures += bad != 0;

    float t_dec = run(ECB_DEC, gpu.data(), back.data(), bytes, iv);
    bool round_trip = back == data;
    printf("  ECB decrypt(encrypt(x)) == x: %s\n", round_trip ? "yes" : "NO");
    failures += !round_trip;

    float t_ctr = run(CTR, data.data(), gpu.data(), bytes, iv);
    bad = 0;
    for (size_t b = 0; b < bytes; b += 16) {
        aes_cpu::ctr_block(rk, iv, b / 16, ref.data());
        for (int i = 0; i < 16; i++) ref[i] ^= data[b + i];
        bad += memcmp(ref.data(), &gpu[b], 16) != 0;
    }
    printf("  CTR: %zu blocks differ from the CPU\n", bad);
    failures += bad != 0;

    double gb = bytes / 1e9;
    printf("\nThroughput (kernel time, GB/s)\n  ECB encrypt %.2f\n  ECB decrypt %.2f\n  CTR         %.2f\n"
           "  CPU reference, one core (byte-oriented, unoptimised): %.3f GB/s\n",
           gb / (t_enc / 1e3), gb / (t_dec / 1e3), gb / (t_ctr / 1e3), gb / cpu_s);
    printf("\n%s\n", failures ? "TESTS FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
