
- Default optimizations: 23.02s
  - sevenlayer_600nm_100M.mco

- Enable USE_TRUE_CACHE (4 copies of A_rz): 55.70s
  - sevenlayer_600nm_100M.true_cache_4.mco

- Enable USE_TRUE_CACHE (8 copies of A_rz): 47.89s
  - sevenlayer_600nm_100M.true_cache_8.mco

- Enable USE_TRUE_CACHE (15 copies of A_rz): 48.79s
  - sevenlayer_600nm_100M.true_cache_15.mco

- Use 32-bit elem for A_rz in smem: 33.13s
  - sevenlayer_600nm_100M.32b_smem.mco

- Cache a 64x96 region of A_rz in smem (4 copies of A_rz): 26.33s
  - sevenlayer_600nm_100M.64_96_4.mco

- Cache a 64x96 region of A_rz in smem (8 copies of A_rz): 27.25s
  - sevenlayer_600nm_100M.64_96_8.mco

- Use 40 regs per thread and 768 threads per block: 23.45s
  - sevenlayer_600nm_100M.40r768t.mco
