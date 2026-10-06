#!/usr/bin/env python3
"""Runs the kernels of src/aes_gpu.cuh on the GPU through NVRTC (CuPy) and
checks them against an independent AES implementation (the `cryptography`
package, backed by OpenSSL). No C++ host compiler is needed.

    pip install cupy-cuda12x cryptography      # or cupy-cuda13x for CUDA 13
    python tests/verify_gpu.py [MiB]

Checks: FIPS-197 and SP 800-38A known answers, then random data in ECB
(encrypt and decrypt) and CTR against OpenSSL, then kernel throughput.
"""
import os
import sys
from pathlib import Path

import cupy as cp
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ROOT = Path(__file__).resolve().parent.parent
THREADS = 256


def expand_key(key: bytes):
    """Encryption and equivalent-inverse-cipher round keys (44 big-endian words each)."""
    sbox = SBOX
    rcon = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
    w = [int.from_bytes(key[4 * i:4 * i + 4], "big") for i in range(4)]
    for i in range(4, 44):
        t = w[i - 1]
        if i % 4 == 0:
            t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF
            t = (sbox[t >> 24] << 24) | (sbox[(t >> 16) & 0xFF] << 16) | (sbox[(t >> 8) & 0xFF] << 8) | sbox[t & 0xFF]
            t ^= rcon[i // 4] << 24
        w.append(w[i - 4] ^ t)

    def xt(a):
        return ((a << 1) ^ (0x1B if a & 0x80 else 0)) & 0xFF

    def mul(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            a = xt(a)
            b >>= 1
        return p

    def inv_mix(x):
        a = [(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF]
        m = [[14, 11, 13, 9], [9, 14, 11, 13], [13, 9, 14, 11], [11, 13, 9, 14]]
        r = [mul(a[0], m[k][0]) ^ mul(a[1], m[k][1]) ^ mul(a[2], m[k][2]) ^ mul(a[3], m[k][3]) for k in range(4)]
        return (r[0] << 24) | (r[1] << 16) | (r[2] << 8) | r[3]

    drk = []
    for r in range(11):
        for c in range(4):
            x = w[4 * (10 - r) + c]
            drk.append(x if r in (0, 10) else inv_mix(x))
    return np.array(w, dtype=np.uint32), np.array(drk, dtype=np.uint32)


SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]


class Gpu:
    def __init__(self):
        src = (ROOT / "src" / "aes_gpu.cuh").read_text()
        self.mod = cp.RawModule(code=src, options=("-std=c++17",))
        self.k_enc = self.mod.get_function("aes128_ecb_encrypt")
        self.k_dec = self.mod.get_function("aes128_ecb_decrypt")
        self.k_ctr = self.mod.get_function("aes128_ctr_xcrypt")
        self.sms = cp.cuda.Device().attributes["MultiProcessorCount"]

    def set_key(self, key: bytes):
        rk, drk = expand_key(key)
        cp.cuda.runtime.memcpy(self.mod.get_global("c_rk").ptr, rk.ctypes.data, rk.nbytes, cp.cuda.runtime.memcpyHostToDevice)
        cp.cuda.runtime.memcpy(self.mod.get_global("c_drk").ptr, drk.ctypes.data, drk.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    def run(self, mode: str, data: bytes, iv: bytes = bytes(16)):
        nblocks = len(data) // 16
        d_in = cp.asarray(np.frombuffer(data, dtype=np.uint8))
        d_out = cp.empty_like(d_in)
        grid = max(1, min((nblocks + THREADS - 1) // THREADS, self.sms * 32))
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        n = np.uint64(nblocks)
        if mode == "enc":
            self.k_enc((grid,), (THREADS,), (d_in, d_out, n))
        elif mode == "dec":
            self.k_dec((grid,), (THREADS,), (d_in, d_out, n))
        else:
            w = [np.uint32(int.from_bytes(iv[4 * i:4 * i + 4], "big")) for i in range(4)]
            self.k_ctr((grid,), (THREADS,), (d_in, d_out, n, *w))
        stop.record()
        stop.synchronize()
        ms = cp.cuda.get_elapsed_time(start, stop)
        return cp.asnumpy(d_out).tobytes(), ms


def openssl(mode: str, key: bytes, data: bytes, iv: bytes = bytes(16)) -> bytes:
    if mode == "ctr":
        c = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    elif mode == "enc":
        c = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    else:
        c = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return c.update(data) + c.finalize()


def main():
    mib = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    gpu = Gpu()
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU: {name} ({gpu.sms} SMs), kernels compiled with NVRTC from src/aes_gpu.cuh\n")
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"  {label:45s} {'PASS' if ok else 'FAIL'}")

    print("Known-answer tests")
    h = bytes.fromhex
    k1 = h("2b7e151628aed2a6abf7158809cf4f3c")
    gpu.set_key(k1)
    check("FIPS-197 Appendix B encrypt", gpu.run("enc", h("3243f6a8885a308d313198a2e0370734"))[0], h("3925841d02dc09fbdc118597196a0b32"))
    check("FIPS-197 Appendix B decrypt", gpu.run("dec", h("3925841d02dc09fbdc118597196a0b32"))[0], h("3243f6a8885a308d313198a2e0370734"))
    pt = h("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710")
    ecb = h("3ad77bb40d7a3660a89ecaf32466ef97f5d3d58503b9699de785895a96fdbaaf43b1cd7f598ece23881b00e3ed0306887b0c785e27e8ad3f8223207104725dd4")
    ctr = h("874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee")
    iv = h("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    check("SP 800-38A F.1.1 ECB encrypt (4 blocks)", gpu.run("enc", pt)[0], ecb)
    check("SP 800-38A F.1.2 ECB decrypt (4 blocks)", gpu.run("dec", ecb)[0], pt)
    check("SP 800-38A F.5.1 CTR encrypt (4 blocks)", gpu.run("ctr", pt, iv)[0], ctr)
    check("SP 800-38A F.5.2 CTR decrypt (4 blocks)", gpu.run("ctr", ctr, iv)[0], pt)
    k2 = h("000102030405060708090a0b0c0d0e0f")
    gpu.set_key(k2)
    check("FIPS-197 Appendix C.1 encrypt", gpu.run("enc", h("00112233445566778899aabbccddeeff"))[0], h("69c4e0d86a7b0430d8cdb78070b4c55a"))
    check("FIPS-197 Appendix C.1 decrypt", gpu.run("dec", h("69c4e0d86a7b0430d8cdb78070b4c55a"))[0], h("00112233445566778899aabbccddeeff"))

    rng = np.random.default_rng(2026)
    data = rng.integers(0, 256, mib << 20, dtype=np.uint8).tobytes()
    key = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
    # the low 64 bits of the counter start just below 2^64, so the carry into the high half is tested
    iv = rng.integers(0, 256, 8, dtype=np.uint8).tobytes() + b"\xff\xff\xff\xff\xff\xff\xff\xf0"
    gpu.set_key(key)
    print(f"\nRandom data: {mib} MiB ({len(data) // 16:,} blocks), random key, counter crossing 2^64")
    enc, t_enc = gpu.run("enc", data)
    check("ECB encrypt equals OpenSSL", enc, openssl("enc", key, data))
    dec, t_dec = gpu.run("dec", enc)
    check("ECB decrypt(encrypt(x)) equals x", dec, data)
    c, t_ctr = gpu.run("ctr", data, iv)
    check("CTR equals OpenSSL", c, openssl("ctr", key, data, iv))

    gb = len(data) / 1e9
    # warm runs for timing (the first launch includes module load)
    t_enc = min(gpu.run("enc", data)[1] for _ in range(3))
    t_dec = min(gpu.run("dec", data)[1] for _ in range(3))
    t_ctr = min(gpu.run("ctr", data, iv)[1] for _ in range(3))
    print(f"\nKernel throughput (best of 3, kernel time only)")
    print(f"  ECB encrypt {gb / (t_enc / 1e3):6.2f} GB/s")
    print(f"  ECB decrypt {gb / (t_dec / 1e3):6.2f} GB/s")
    print(f"  CTR         {gb / (t_ctr / 1e3):6.2f} GB/s")
    print(f"\n{'all tests passed' if not fails else 'TESTS FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
