# Parallel Encryption with CUDA

This project implements parallel encryption and decryption algorithms using **CUDA C**, leveraging GPU architecture for high-performance cryptographic operations.

The implementation demonstrates how symmetric key encryption can be efficiently parallelized across many threads, significantly improving performance compared to CPU-based methods.

## 📂 Contents

- `encrypt.cu` – CUDA kernel for data encryption
- `decrypt.cu` – CUDA kernel for data decryption

## 🔒 Algorithm Overview

The encryption algorithm is a simplified block cipher operating on data chunks using XOR-based operations in parallel threads.

## 🛠️ Technologies Used

- CUDA C / C++
- NVIDIA GPU
- NVCC compiler (from CUDA Toolkit)

## 🚀 How to Compile & Run

You need a CUDA-compatible NVIDIA GPU and the **CUDA Toolkit** installed.

### 🔧 Compilation

```bash
nvcc encrypt.cu -o encrypt
nvcc decrypt.cu -o decrypt
