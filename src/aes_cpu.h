// AES-128 on the CPU: key expansion for the GPU kernels and a byte-oriented
// reference implementation (FIPS-197) used to check the GPU results.
#ifndef AES_CPU_H
#define AES_CPU_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

namespace aes_cpu {

static const uint8_t SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static inline uint8_t xtime(uint8_t a) { return (uint8_t)((a << 1) ^ ((a & 0x80) ? 0x1b : 0)); }
static inline uint8_t mul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int i = 0; i < 8; i++) { if (b & 1) p ^= a; a = xtime(a); b >>= 1; }
    return p;
}
static inline uint8_t inv_sbox(uint8_t x) {
    static uint8_t inv[256];
    static bool ready = false;
    if (!ready) { for (int i = 0; i < 256; i++) inv[SBOX[i]] = (uint8_t)i; ready = true; }
    return inv[x];
}

// Encryption round keys: 44 big-endian words (FIPS-197 key expansion).
static inline void expand_key(const uint8_t key[16], uint32_t rk[44]) {
    static const uint8_t RCON[11] = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
    for (int i = 0; i < 4; i++)
        rk[i] = ((uint32_t)key[4 * i] << 24) | ((uint32_t)key[4 * i + 1] << 16) | ((uint32_t)key[4 * i + 2] << 8) | key[4 * i + 3];
    for (int i = 4; i < 44; i++) {
        uint32_t t = rk[i - 1];
        if (i % 4 == 0) {
            t = (t << 8) | (t >> 24);
            t = ((uint32_t)SBOX[t >> 24] << 24) | ((uint32_t)SBOX[(t >> 16) & 0xff] << 16) |
                ((uint32_t)SBOX[(t >> 8) & 0xff] << 8) | SBOX[t & 0xff];
            t ^= (uint32_t)RCON[i / 4] << 24;
        }
        rk[i] = rk[i - 4] ^ t;
    }
}

static inline uint32_t inv_mix_word(uint32_t w) {
    uint8_t a0 = w >> 24, a1 = w >> 16, a2 = w >> 8, a3 = (uint8_t)w;
    return ((uint32_t)(mul(a0, 14) ^ mul(a1, 11) ^ mul(a2, 13) ^ mul(a3, 9)) << 24) |
           ((uint32_t)(mul(a0, 9) ^ mul(a1, 14) ^ mul(a2, 11) ^ mul(a3, 13)) << 16) |
           ((uint32_t)(mul(a0, 13) ^ mul(a1, 9) ^ mul(a2, 14) ^ mul(a3, 11)) << 8) |
           (uint32_t)(mul(a0, 11) ^ mul(a1, 13) ^ mul(a2, 9) ^ mul(a3, 14));
}

// Round keys of the equivalent inverse cipher (FIPS-197 section 5.3.5),
// in the order the decryption kernel uses them.
static inline void expand_decrypt_key(const uint32_t rk[44], uint32_t drk[44]) {
    for (int r = 0; r <= 10; r++)
        for (int c = 0; c < 4; c++) {
            uint32_t w = rk[4 * (10 - r) + c];
            drk[4 * r + c] = (r == 0 || r == 10) ? w : inv_mix_word(w);
        }
}

// Byte-oriented reference: straight from the FIPS-197 pseudocode.
static inline void encrypt_block_ref(const uint32_t rk[44], const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    memcpy(s, in, 16);
    auto add_rk = [&](int r) {
        for (int c = 0; c < 4; c++)
            for (int j = 0; j < 4; j++) s[4 * c + j] ^= (uint8_t)(rk[4 * r + c] >> (24 - 8 * j));
    };
    add_rk(0);
    for (int r = 1; r <= 10; r++) {
        for (int i = 0; i < 16; i++) s[i] = SBOX[s[i]];
        uint8_t t[16];
        for (int c = 0; c < 4; c++)
            for (int j = 0; j < 4; j++) t[4 * c + j] = s[4 * ((c + j) % 4) + j];
        memcpy(s, t, 16);
        if (r != 10)
            for (int c = 0; c < 4; c++) {
                uint8_t a0 = s[4 * c], a1 = s[4 * c + 1], a2 = s[4 * c + 2], a3 = s[4 * c + 3];
                s[4 * c] = mul(a0, 2) ^ mul(a1, 3) ^ a2 ^ a3;
                s[4 * c + 1] = a0 ^ mul(a1, 2) ^ mul(a2, 3) ^ a3;
                s[4 * c + 2] = a0 ^ a1 ^ mul(a2, 2) ^ mul(a3, 3);
                s[4 * c + 3] = mul(a0, 3) ^ a1 ^ a2 ^ mul(a3, 2);
            }
        add_rk(r);
    }
    memcpy(out, s, 16);
}

static inline void decrypt_block_ref(const uint32_t rk[44], const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    memcpy(s, in, 16);
    auto add_rk = [&](int r) {
        for (int c = 0; c < 4; c++)
            for (int j = 0; j < 4; j++) s[4 * c + j] ^= (uint8_t)(rk[4 * r + c] >> (24 - 8 * j));
    };
    add_rk(10);
    for (int r = 9; r >= 0; r--) {
        uint8_t t[16];
        for (int c = 0; c < 4; c++)
            for (int j = 0; j < 4; j++) t[4 * ((c + j) % 4) + j] = s[4 * c + j];
        memcpy(s, t, 16);
        for (int i = 0; i < 16; i++) s[i] = inv_sbox(s[i]);
        add_rk(r);
        if (r != 0)
            for (int c = 0; c < 4; c++) {
                uint8_t a0 = s[4 * c], a1 = s[4 * c + 1], a2 = s[4 * c + 2], a3 = s[4 * c + 3];
                s[4 * c] = mul(a0, 14) ^ mul(a1, 11) ^ mul(a2, 13) ^ mul(a3, 9);
                s[4 * c + 1] = mul(a0, 9) ^ mul(a1, 14) ^ mul(a2, 11) ^ mul(a3, 13);
                s[4 * c + 2] = mul(a0, 13) ^ mul(a1, 9) ^ mul(a2, 14) ^ mul(a3, 11);
                s[4 * c + 3] = mul(a0, 11) ^ mul(a1, 13) ^ mul(a2, 9) ^ mul(a3, 14);
            }
    }
    memcpy(out, s, 16);
}

// CTR keystream block for block index i (128-bit counter addition, SP 800-38A).
static inline void ctr_block(const uint32_t rk[44], const uint8_t iv[16], uint64_t i, uint8_t ks[16]) {
    uint8_t ctr[16];
    memcpy(ctr, iv, 16);
    uint64_t lo = 0, hi = 0;
    for (int k = 0; k < 8; k++) { hi = (hi << 8) | ctr[k]; lo = (lo << 8) | ctr[8 + k]; }
    uint64_t l2 = lo + i;
    if (l2 < lo) hi++;
    for (int k = 7; k >= 0; k--) { ctr[k] = (uint8_t)hi; hi >>= 8; ctr[8 + k] = (uint8_t)l2; l2 >>= 8; }
    encrypt_block_ref(rk, ctr, ks);
}

}  // namespace aes_cpu

#endif
