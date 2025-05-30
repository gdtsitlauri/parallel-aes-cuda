/******************************************************************************
 * AES-128 ENCRYPTION USING CUDA
 * 
 * Single-block demonstration:
 * - Encrypts one 128-bit block (plaintext) with a 128-bit key.
 ******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// AES block size in bytes
#define AES_BLOCK_SIZE 16

// -----------------------------------------------------------------------------
// Device constants
// -----------------------------------------------------------------------------
__constant__ uint8_t d_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5, 0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0, 0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc, 0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a, 0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0, 0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b, 0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85, 0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5, 0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17, 0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88, 0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c, 0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9, 0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6, 0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e, 0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94, 0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68, 0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__constant__ uint8_t d_Rcon[11] = {
    0x00, // Rcon[0] is unused
    0x01, 0x02, 0x04, 0x08,
    0x10, 0x20, 0x40, 0x80,
    0x1B, 0x36
};

// -----------------------------------------------------------------------------
// Device utility functions
// -----------------------------------------------------------------------------
__device__ uint8_t galois_mul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    uint8_t hi_bit_set;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        hi_bit_set = (a & 0x80);
        a <<= 1;
        if (hi_bit_set) {
            // x^8 + x^4 + x^3 + x + 1 = 0x1B
            a ^= 0x1B;
        }
        b >>= 1;
    }
    return p;
}

__device__ void add_round_key(uint8_t state[4][4], uint8_t round_key[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] ^= round_key[i][j];
        }
    }
}

__device__ void sub_bytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] = d_sbox[state[i][j]];
        }
    }
}

__device__ void shift_rows(uint8_t state[4][4]) {
    uint8_t temp;

    // Row 1: shift left by 1
    temp = state[1][0];
    state[1][0] = state[1][1];
    state[1][1] = state[1][2];
    state[1][2] = state[1][3];
    state[1][3] = temp;

    // Row 2: shift left by 2
    temp = state[2][0];
    state[2][0] = state[2][2];
    state[2][2] = temp;
    temp = state[2][1];
    state[2][1] = state[2][3];
    state[2][3] = temp;

    // Row 3: shift left by 3 (equivalent to shift right by 1)
    temp = state[3][3];
    state[3][3] = state[3][2];
    state[3][2] = state[3][1];
    state[3][1] = state[3][0];
    state[3][0] = temp;
}

__device__ void mix_columns(uint8_t state[4][4]) {
    uint8_t tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[0] = galois_mul(state[0][i], 2) ^ galois_mul(state[1][i], 3) ^ state[2][i] ^ state[3][i];
        tmp[1] = state[0][i] ^ galois_mul(state[1][i], 2) ^ galois_mul(state[2][i], 3) ^ state[3][i];
        tmp[2] = state[0][i] ^ state[1][i] ^ galois_mul(state[2][i], 2) ^ galois_mul(state[3][i], 3);
        tmp[3] = galois_mul(state[0][i], 3) ^ state[1][i] ^ state[2][i] ^ galois_mul(state[3][i], 2);

        state[0][i] = tmp[0];
        state[1][i] = tmp[1];
        state[2][i] = tmp[2];
        state[3][i] = tmp[3];
    }
}

__device__ void key_expansion(uint8_t key[4][4], uint8_t round_keys[11][4][4]) {
    uint32_t w[44];
    // Copy the original key into the first 4 words
    for (int i = 0; i < 4; i++) {
        w[i] = ((uint32_t)key[0][i] << 24) |
               ((uint32_t)key[1][i] << 16) |
               ((uint32_t)key[2][i] << 8)  |
               ((uint32_t)key[3][i]);
    }

    // Expand the key
    for (int i = 4; i < 44; i++) {
        uint32_t temp = w[i - 1];
        if ((i % 4) == 0) {
            // RotWord
            temp = (temp << 8) | (temp >> 24);
            // SubWord
            temp = ((uint32_t)d_sbox[(temp >> 24) & 0xFF] << 24) |
                   ((uint32_t)d_sbox[(temp >> 16) & 0xFF] << 16) |
                   ((uint32_t)d_sbox[(temp >> 8)  & 0xFF] << 8)  |
                   (uint32_t)d_sbox[temp & 0xFF];
            // Rcon
            temp ^= ((uint32_t)d_Rcon[i / 4] << 24);
        }
        w[i] = w[i - 4] ^ temp;
    }

    // Store expanded keys in round_keys
    for (int round = 0; round < 11; round++) {
        for (int i = 0; i < 4; i++) {
            uint32_t word = w[round * 4 + i];
            round_keys[round][0][i] = (word >> 24) & 0xFF;
            round_keys[round][1][i] = (word >> 16) & 0xFF;
            round_keys[round][2][i] = (word >> 8) & 0xFF;
            round_keys[round][3][i] = word & 0xFF;
        }
    }
}

__device__ void aes_encrypt_device(const uint32_t input[4], const uint32_t key[4], uint32_t output[4]) {
    // Convert input into state
    uint8_t state[4][4];
    for (int i = 0; i < 4; i++) {
        state[0][i] = (input[i] >> 24) & 0xFF;
        state[1][i] = (input[i] >> 16) & 0xFF;
        state[2][i] = (input[i] >> 8)  & 0xFF;
        state[3][i] =  input[i]        & 0xFF;
    }

    // Convert key into 4x4
    uint8_t key_state[4][4];
    for (int i = 0; i < 4; i++) {
        key_state[0][i] = (key[i] >> 24) & 0xFF;
        key_state[1][i] = (key[i] >> 16) & 0xFF;
        key_state[2][i] = (key[i] >> 8)  & 0xFF;
        key_state[3][i] =  key[i]        & 0xFF;
    }

    // Generate round keys
    uint8_t round_keys[11][4][4];
    key_expansion(key_state, round_keys);

    // Initial AddRoundKey
    add_round_key(state, round_keys[0]);

    // Rounds 1 to 9
    for (int round = 1; round <= 9; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, round_keys[round]);
    }

    // Final Round
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, round_keys[10]);

    // Copy state back to output
    for (int i = 0; i < 4; i++) {
        output[i] = (state[0][i] << 24) |
                    (state[1][i] << 16) |
                    (state[2][i] << 8)  |
                     state[3][i];
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel
// -----------------------------------------------------------------------------
__global__ void aes_encrypt_kernel(const uint32_t *d_input, 
                                   const uint32_t *d_key, 
                                   uint32_t *d_output)
{
    // For a single block demonstration, we assume thread 0 does the work.
    // For multiple blocks, index with threadIdx, blockIdx.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        aes_encrypt_device(d_input, d_key, d_output);
    }
}

int main() {
    // Example data
    uint32_t h_plaintext[4] = {
        0x3243f6a8, 
        0x885a308d, 
        0x313198a2, 
        0xe0370734
    };
    uint32_t h_key[4] = {
        0x2b7e1516, 
        0x28aed2a6, 
        0xabf71588, 
        0x09cf4f3c
    };
    uint32_t h_ciphertext[4]; // output
    uint32_t h_expected[4] = {
        0x3925841d, 
        0x02dc09fb, 
        0xdc118597, 
        0x196a0b32
    };

    // Allocate device memory
    uint32_t *d_plaintext, *d_key, *d_ciphertext;
    cudaMalloc((void**)&d_plaintext,   4 * sizeof(uint32_t));
    cudaMalloc((void**)&d_key,         4 * sizeof(uint32_t));
    cudaMalloc((void**)&d_ciphertext,  4 * sizeof(uint32_t));

    // Copy data from host to device
    cudaMemcpy(d_plaintext,  h_plaintext,  4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key,        h_key,        4 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel (1 block, 1 thread)
    aes_encrypt_kernel<<<1, 1>>>(d_plaintext, d_key, d_ciphertext);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_ciphertext, d_ciphertext, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_plaintext);
    cudaFree(d_key);
    cudaFree(d_ciphertext);

    // Print results
    std::cout << "Encrypted ciphertext: ";
    for (int i = 0; i < 4; i++) {
        printf("%08x ", h_ciphertext[i]);
    }
    std::cout << std::endl;

    // Check
    bool match = true;
    for (int i = 0; i < 4; i++) {
        if (h_ciphertext[i] != h_expected[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Test Passed: The ciphertext matches the expected value!" << std::endl;
    } else {
        std::cout << "Test Failed: The ciphertext does NOT match the expected value." << std::endl;
    }

    return 0;
}
