// Checks the CPU reference in src/aes_cpu.h (used by aes_cuda.cu to verify the GPU)
// against FIPS-197 and SP 800-38A, without a GPU:  g++ -std=c++17 tests/test_cpu_ref.cpp
#include <cstdio>
#include <cstring>
#include "../src/aes_cpu.h"

static void hex(const char *s, uint8_t *o, size_t n) { for (size_t i = 0; i < n; i++) sscanf(s + 2 * i, "%2hhx", &o[i]); }

int main() {
    int fails = 0;
    auto check = [&](const char *name, const uint8_t *a, const uint8_t *b) {
        bool ok = !memcmp(a, b, 16);
        printf("  %-36s %s\n", name, ok ? "PASS" : "FAIL");
        fails += !ok;
    };
    uint8_t k[16], p[16], c[16], o[16], iv[16];
    uint32_t rk[44];
    hex("000102030405060708090a0b0c0d0e0f", k, 16); aes_cpu::expand_key(k, rk);
    hex("00112233445566778899aabbccddeeff", p, 16); hex("69c4e0d86a7b0430d8cdb78070b4c55a", c, 16);
    aes_cpu::encrypt_block_ref(rk, p, o); check("FIPS-197 C.1 encrypt", o, c);
    aes_cpu::decrypt_block_ref(rk, c, o); check("FIPS-197 C.1 decrypt", o, p);
    hex("2b7e151628aed2a6abf7158809cf4f3c", k, 16); aes_cpu::expand_key(k, rk);
    hex("3243f6a8885a308d313198a2e0370734", p, 16); hex("3925841d02dc09fbdc118597196a0b32", c, 16);
    aes_cpu::encrypt_block_ref(rk, p, o); check("FIPS-197 B encrypt", o, c);
    hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff", iv, 16);
    hex("6bc1bee22e409f96e93d7e117393172a", p, 16); hex("874d6191b620e3261bef6864990db6ce", c, 16);
    aes_cpu::ctr_block(rk, iv, 0, o); for (int i = 0; i < 16; i++) o[i] ^= p[i]; check("SP 800-38A F.5.1 block 1", o, c);
    hex("ae2d8a571e03ac9c9eb76fac45af8e51", p, 16); hex("9806f66b7970fdff8617187bb9fffdff", c, 16);
    aes_cpu::ctr_block(rk, iv, 1, o); for (int i = 0; i < 16; i++) o[i] ^= p[i]; check("SP 800-38A F.5.1 block 2 (carry)", o, c);
    printf("%s\n", fails ? "FAILED" : "all CPU reference tests passed");
    return fails != 0;
}
