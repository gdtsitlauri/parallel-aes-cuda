# Parallel AES Cryptography with CUDA

This project implements the full **AES (Advanced Encryption Standard)** algorithm for both encryption and decryption using **CUDA C**. It demonstrates how symmetric block ciphers like AES can be parallelized and executed on the GPU for significant speedups.

The code was tested and executed successfully using **Visual Studio Code** and an **NVIDIA GTX 1650** GPU.

## 📂 Contents

- `encrypt.cu` – CUDA source file for AES encryption
- `decrypt.cu` – CUDA source file for AES decryption

## 🔐 Algorithm Overview

- AES-128 standard (Rijndael cipher)
- 128-bit block size
- 10 rounds with:
  - SubBytes
  - ShiftRows
  - MixColumns
  - AddRoundKey
- Key expansion (key scheduling) fully implemented
- Parallel processing of multiple data blocks using CUDA threads

## 🧠 CUDA Features Used

- Thread and block-level parallelism
- Device memory management
- Shared memory access (optional for optimization)
- NVCC compilation targeting compute capability compatible with GTX 1650

## 🛠️ Technologies

- CUDA C/C++
- NVIDIA GPU (GTX 1650 or higher)
- Visual Studio Code (tested)
- NVCC (NVIDIA CUDA Compiler)

## 🚀 How to Compile and Run

### 1. Compile with `nvcc`

```bash
nvcc encrypt.cu -o encrypt
nvcc decrypt.cu -o decrypt
