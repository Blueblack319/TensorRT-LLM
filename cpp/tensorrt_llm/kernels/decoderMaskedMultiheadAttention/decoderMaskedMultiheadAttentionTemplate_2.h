/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
// [ ] Maybe?
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include <assert.h>
#include <float.h>
#include <type_traits>
// CHECKLIST
#include "tensorrt_llm/common/cudaUtils.h"

// Multi-block mmha kernel can only be selected when CUDA >= 11.7
#if (CUDART_VERSION >= 11070)
#define ENABLE_MULTI_BLOCK_OPTION
#endif

#ifdef ENABLE_MULTI_BLOCK_OPTION
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#endif // ENABLE_MULTI_BLOCK_OPTION

namespace tensorrt_llm
{
namespace kernels
{

// Use HMMA to compute with FP16/BF16 inputs and FP32 accumulators.
// #define MMHA_USE_HMMA

// Pre-scale Q or P to reduce number of instructions for dequantizing KV cache.
// If you notice a decrease in accuracy when the fp8 kv cache is enabled,
//  consider disabling the two flags.
#ifdef ENABLE_FP8
// Apply the FP8 scaling to Q instead of K.
#define MMHA_FP8_SCALE_Q_INSTEAD_OF_K
// Apply the FP8 scaling to P instead of V.
#define MMHA_FP8_SCALE_P_INSTEAD_OF_V
#endif // !defined ENABLE_FP8

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
#define MMHA_USE_FP32_ACCUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACCUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACCUM_FOR_OUT)
 // Does not seem to improve the accuracy
 //#define MMHA_USE_FP32_ACCUM_FOR_LOGITS
#endif

namespace mmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.
//
// The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
// 256 threads per block to maximum occupancy and performance.
//
// Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
// compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
// cache buffer helps with memory accesses and contains keys with bias.
//
// The layout of the cache buffer for the keys/values is [B, H, L, Dh]
// where the fastest moving dimension (contiguous data) is the rightmost one.
// Contiguous threads will read one hidden_dimension per LDG unless we need more than 32 threads.
//
// The different kernels use 1 ~ 32 threads per key (THREADS_PER_KEY). The size of the LDGs
// is always 16bytes (8 bytes for 8bit cache). Each thread sums Dh / THREADS_PER_KEY elements. At
// the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
// HMMA instruction (Tensor Core). Each Q * K^T value is stored in shared memory in FP32.
//
// After that loop, a parallel softmax is computed across the different Q * K^T values stored in
// shared memory.
//
// The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
// timesteps are computed by loop iteration. As with the keys, the values are read from a cache
// except for the current timestep. The layout of the cache buffer for the values is same as the key,
// which is [B, H, L, Dh].
//
// Note that we have remapped key layout to make sure it shares the same pattern as value [B, H, L, Dh].
// It helps coalescing memory access, and reducing register pressure.

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The type of the inputs. Supported types: float, uint16_t, nv_bfloat16.
    typename T,
    // The type of the cache.
    typename Tcache,
    // Type of struct containing KV cache
    typename KVCacheBuffer,
    // The hidden dimension per head.
    unsigned Dh,
    // The number of threads in a threadblock.
    unsigned THREADS_PER_BLOCK,
    // Whether cross attention is enabled
    bool DO_CROSS_ATTENTION,
    // Whether has beams.
    bool HAS_BEAMS,
    // Whether enable multi-block mode for long-sequence-length.
    bool DO_MULTI_BLOCK = false,
    // The number of threads per key.
    unsigned THREADS_PER_KEY = threads_per_key<T, dh_max(Dh)>(),
    // The number of threads per value.
    unsigned THREADS_PER_VALUE = threads_per_value<T>(dh_max(Dh)),
    // The unroll factor for loading from K cache.
    // Set it default to 4 for higher occupancy (by reducing registers usage).
    unsigned K_LOOP_UNROLL = 4,
    // The unroll factor for loading from V cache.
    unsigned V_LOOP_UNROLL = 8>
__global__ void masked_multihead_attention_kernel_2(
    Multihead_attention_params<T, DO_CROSS_ATTENTION> params, KVCacheBuffer kvCacheBuffer)
{

    using Tk = typename kernel_type_t<T>::Type;
    // Use 8bit cache.
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;
    // FP8 KV Cache.
    static constexpr bool FP8_KV_CACHE = std::is_same<Tcache, __nv_fp8_e4m3>::value;
    // INT8 KV Cache.
    static constexpr bool INT8_KV_CACHE = std::is_same<Tcache, int8_t>::value;

    // The size of a warp.
    constexpr unsigned WARP_SIZE{32};
    // The number of warps in a threadblock.
    constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

    // The maximum hidden size per head.
    constexpr auto Dh_MAX = dh_max(Dh);
    constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
    static_assert(Dh_MAX >= WARP_SIZE);
    static_assert(Dh_MAX >= Dh);

    // The maximum sequence length in the cyclic kv_cache, i.e., an upper bound on L.
    // Note that the maximum sequence length supported by the model might be greater than this.
    // Note max_attention_window_size is maximum of cyclic_attention_window_size among all layers.
    // By default, you can assume that they are the same.
    const auto cyclic_kv_cache_len = static_cast<unsigned>(params.cyclic_attention_window_size);
    // The current timestep (including paddings).
    // It is only used to calculate the smem stride.
    const auto timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep);

#ifdef ENABLE_MULTI_BLOCK_OPTION
    constexpr bool MULTI_BLOCK_FLAG = DO_MULTI_BLOCK;
#else
    constexpr bool MULTI_BLOCK_FLAG = false;
#endif

    // Use smem_size_in_bytes (above) to determine the amount of shared memory.
    extern __shared__ char smem_[];

    // The shared memory for the Q*K^T values and partial logits in softmax.
    auto qk_smem = reinterpret_cast<float*>(smem_);

    __shared__ float qk_current_smem[1];

    // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACCUM_FOR_LOGITS
    if (sizeof(Tk) != 4)
    {
        const auto max_timesteps = DO_CROSS_ATTENTION ? cyclic_kv_cache_len : min(timestep, cyclic_kv_cache_len);
        logits_smem_ += divUp(max_timesteps + 1, 4u) * 16;
    }
    Tk* logits_smem = reinterpret_cast<Tk*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    __shared__ Tk logits_current_smem[1];

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    Tk* out_smem = reinterpret_cast<Tk*>(smem_);

    // The shared memory buffers for the block-wide reductions. One for max, one for sum.
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec_m = typename Qk_vec_m_<T, Dh_MAX>::Type; // with memory-used precision
    using Qk_vec_k = typename Qk_vec_k_<T, Dh_MAX>::Type; // with kernel-used precision
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using Qk_vec_accum = typename Qk_vec_accum_fp32_<Qk_vec_k>::Type;
#else
    using Qk_vec_accum = Qk_vec_k;
#endif

    // Make sure the hidden dimension per head is a multiple of the number of threads per key.
    static_assert(Dh_MAX % THREADS_PER_KEY == 0); // trivially satisfied since THREADS_PER_KEY in {1, 2, 4}

    // The number of elements per vector.
    // Each thread will handle 16 bytes.
    constexpr int K_VEC_SIZE = 16u / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % K_VEC_SIZE == 0);
    // The type of queries and keys for the math in the Q*K^T product.
    using K_vec_k = typename K_vec_k_<T, K_VEC_SIZE>::Type;
    // Only used when key cache is quantized to 8 bits.
    using K_vec_m = typename packed_type<Tcache, num_elems<K_vec_k>::value>::type;
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename Qk_vec_accum_fp32_<K_vec_k>::Type;
#else
    using K_vec_accum = K_vec_k;
#endif

    // Use alignment for safely casting the shared buffers as Qk_vec_k and K_vec_k.
    // Shared memory to store Q inputs.
    __shared__ __align__(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk q_smem[Dh_MAX];
    __shared__ __align__(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk k_smem[Dh_MAX];

    // Make sure the hidden dimension per head is a multiple of the number of threads per value.
    static_assert(Dh_MAX % THREADS_PER_VALUE == 0); // trivially satisfied since THREADS_PER_VALUE == Dh_MAX / p

    // The number of elements per vector.
    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
    // Only used when value cache is quantized to 8 bits.
    using V_vec_m = typename packed_type<Tcache, num_elems<V_vec_k>::value>::type;
    static_assert(V_VEC_SIZE == sizeof(V_vec_k) / sizeof(T));

    // This could be one of the reasons to have a separate kernel for cross attention
    constexpr auto bias_smem_size = DO_CROSS_ATTENTION ? Dh_MAX : 1u;
    __shared__ __align__(mmha::const_max(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k)), sizeof(V_vec_k)))
        Tk bias_smem[bias_smem_size];

    // The number of elements per vector.
    constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0);
    // We will use block wide reduction if needed
    // The number of vectors per Dh_MAX.
    constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
    static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

    // The batch/beam idx
    const auto batch_beam_idx = blockIdx.y;
    if (params.finished != nullptr && params.finished[batch_beam_idx])
    {
        return;
    }

    // The head.
    const unsigned hi{blockIdx.x};
    // The head index of keys and values adjusted for MQA/GQA.
    const int qhead_per_kv{params.num_heads / params.num_kv_heads};
    const unsigned hi_kv{hi / qhead_per_kv};
    // The number of heads.
    const auto num_heads = static_cast<unsigned>(params.num_heads);
    // The number of heads for keys and values adjusted for MQA/GQA.
    const auto num_heads_kv = static_cast<unsigned>(params.num_kv_heads);

    // The thread in the block.
    const unsigned tidx{threadIdx.x};

    // The column tile along L dimension on K^T -- noted as T_c in flash-attention paper
    const unsigned c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};

    // Indicate if we need to compute the K/V cache element (add KV bias, IA3, RoPE, etc.) and update the cache.
    // For Self-Attention, it's always required.
    // For Cross-Attention, as everything is pre-computed,
    // in the context phase of the encoder, it's not needed in that kernel.
    // Therefore, HANDLE_KV is !DO_CROSS_ATTENTION and irrelevant of timestep.
    static constexpr bool HANDLE_KV{!DO_CROSS_ATTENTION};

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    // [x] Never used
    // float qk = 0.0F;

    // Do we have a relative attention bias?
    bool has_relative_attention_bias = params.relative_attention_bias != nullptr;
    // Compute relative attention bias on the fly, with relative attention table [head_num/TP, num_buckets] passed in.
    // num_buckets passed as relative_attention_bias_stride, max_distance passed as params.max_distance
    // this is a common optimization for both self attention and cross attention
    const bool implicit_rel_attn_bias = params.max_distance != 0 && has_relative_attention_bias;
    int relative_attention_bias_stride
        = params.relative_attention_bias_stride; // num_buckets might be modified below, save it beforehand
    int max_distance = params.max_distance;

    // The actual sequence length excluding the paddings.
    // minus 1 because it includes the current timestep while tlength denotes the kv cache length.
    const int tlength = DO_CROSS_ATTENTION
        ? params.memory_length_per_sample[batch_beam_idx] - 1
        : (params.length_per_sample ? (params.length_per_sample[batch_beam_idx] - 1) : static_cast<int>(timestep));
    // We will use cyclic kv cache when it exceeds the limit.
    // The length position for storing new key and value.
    const int cyclic_tlength = tlength % cyclic_kv_cache_len;
    // The actual kv cache length.
    // tlength is the past length actually.
    const int kv_loop_length = min(tlength, cyclic_kv_cache_len);
    // The context length for beam searching optimization (all points to beam 0).
    // TODO: with cyclic kv cache, we set it 0 for now (will optimize in the future)
    // as context kv cache might be overwritten by the new kv cache
    const int beam0_context_length
        = HAS_BEAMS && tlength > cyclic_kv_cache_len ? 0 : params.input_lengths[batch_beam_idx];

    // The offset in the Q and K buffer also accounts for the batch.
    // [x] Never used
    // const auto qk_vec_idx = tidx * QK_VEC_SIZE;
    // [x] Never used
    // const auto is_valid_qk_vec = qk_vec_idx < Dh;

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // CHECKLIST: Loading Q and K with Quantization
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    const bool load_qkv_quant = params.qkv_scale_quant_orig != nullptr;
    const bool write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

    // [x] Define varaibles

    // Quant/Dequant scales for 8bits kv cache.
    using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
    T_scale kv_scale_orig_quant, kv_scale_quant_orig;
    const float kv_scale_quant_orig_f = (ENABLE_8BITS_CACHE ? params.kv_scale_quant_orig[0] : 1.0f);
    convert_from_float(&kv_scale_quant_orig, kv_scale_quant_orig_f);
    convert_from_float(&kv_scale_orig_quant, (ENABLE_8BITS_CACHE ? params.kv_scale_orig_quant[0] : 1.0f));

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // CHECKLIST: Applying Position Embeddings and Biases (B)
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // The width of the beam.
    const auto beam_width = static_cast<unsigned>(params.beam_width);
    // The batch idx.
    const int batch_idx = batch_beam_idx / beam_width;
    // Do we apply IA3?
    const bool do_ia3 = HANDLE_KV && params.ia3_tasks != nullptr;
    // Compute the IA3 task. One per batch index.
    const auto ia3_ti_hi = do_ia3
        ? tensorrt_llm::common::flat_index2(static_cast<unsigned>(params.ia3_tasks[batch_idx]), hi, num_heads)
        : 0;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CHECKLIST: Dot product btw Q and K^T from KV-cache

    constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

    // The positions of the cache buffer (for this B * H) and the vector within that chunk associated with this
    // thread.
    const auto k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

    // The number of vectors per thread.
    constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
    static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);
    // [x] Never used
    // The number of timesteps loaded per iteration, i.e., (THREADS_PER_BLOCK * THREADS_PER_BLOCK) / 256 <= 256
    // constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
    // The number of keys per warp.
    constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
    // The number of unrolled keys per warp.
    constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
    // [x] Never used
    // The number of unrolled keys per ieration.
    // constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

    // Base pointer for the row of pointers to k cache blocks
    void** k_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::K_IDX, batch_beam_idx));

    const auto timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.

    // TODO_:  Need to consider the case that 'context_length' becomes 'beam0_context_length' if we need to more
    // generalize
    const int context_length
        = DO_CROSS_ATTENTION ? kv_loop_length : (HAS_BEAMS ? beam0_context_length : kv_loop_length);
    // Clarifications:
    // - in self attn, input_length is input text length, tlength is current timestep
    // - in cross attn, input_length is *decoder* input length (usually 1), tlength is *encoder* input context length
    // - in beam search, since the cache during generation is organized differently, the following KV compute needs
    // split into context cache compute and generation cache compute
    // - for self attn, no-beam search: entire cache can be treated as context cache --> context_length = tlength
    // - for self attn, beam search: cache of input text length is context cache, other are generation cache -->
    // context_length = input_length
    // - for cross attn, no-beam/beam search: cache length is fixed, not differ context/generation cache -->
    // context_length = tlength Suggestion: we could have a flag HANDLE_GEN_CACHE

    const auto context_ti_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP
        : divUp(static_cast<unsigned>(context_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

    // The generation ti_end.
    const auto generation_ti_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, K_PER_WARP) * K_PER_WARP
        : divUp(static_cast<unsigned>(kv_loop_length), K_PER_WARP) * K_PER_WARP;

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    // Note max_attention_window_size is maximum of cyclic_attention_window_size among all layers.
    // By default, you can assume that they are the same.
    const auto bi_seq_len_offset = static_cast<std::size_t>(batch_beam_idx) * params.max_attention_window_size;
    // Beam indices are based on the max_attention_window_size while each layer may have different
    // cyclic_attention_window_size So we need to rebuild the beam_indices if max_attention_window_size is not equal to
    // cyclic_attention_window_size.
    const int* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    const auto c_tile_times_timesteps_per_block = c_tile * timesteps_per_block; // 0 if !MULTI_BLOCK_FLAG

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Key cache loops for dot(Q, K).

    // Is it the leader?
    const bool is_leader = Qk_dot<T, THREADS_PER_KEY>::is_leader(tidx);

    // CHECKLIST: Split-Point======================================================================================

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // [x] Restore the qk_values and qk_max_value for each head

    // [ ] If this code perform incorrectly, let each thread takes qk_max from qk_max_values.
    const auto lane = tidx % WARP_SIZE;
    qk_max = lane == 0 ? params.qk_max_values[hi] : 0;
    // Broadcast to all the threads in the warp
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    // [x] Restore the qk_smem from params.qk_values
    const auto max_attention_window_size = params.max_attention_window_size;
    const int qk_values_offset = hi * max_attention_window_size;

    // if (tidx <= kv_loop_length)
    // {
#pragma unroll
    for (int out_i = tidx; out_i <= kv_loop_length; out_i += THREADS_PER_BLOCK)
    {
        qk_smem[out_i] = params.qk_values[qk_values_offset + out_i];
    }
    // }
    __syncthreads();

    // TODO_: Need to restore 'qk_current_smem' if we consider 'MULTI_BLOCK_FLAG'

    // [x] Restore the qk_values and qk_max_value for each head
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // CHECKLIST: Compute the logits and start the sum.

    // Compute the logits and start the sum.
    float sum = 0.f;

    // Each thread will handle one float (either qk_smem/logit).
    const int logit_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= logit_loop_end; ti += THREADS_PER_BLOCK)
    {

        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        // For single-block mode, we don't need the mask since it has been skipped.
        if (!MULTI_BLOCK_FLAG)
        {
            float logit = __expf(qk_smem[time_now] - qk_max);
            sum += logit;
            qk_smem[time_now] = logit;
        }
        else
        {
            // Not supported yet: multi-block mode with FP8_MHA
            if (time_now < kv_loop_length && ti != timesteps_per_block)
            {
                float logit = __expf(qk_smem[ti] - qk_max);
                sum += logit;
                qk_smem[ti] = logit;
            }
            else if (time_now == kv_loop_length)
            {
                float logit = __expf(qk_current_smem[0] - qk_max);
                sum += logit;
                qk_current_smem[0] = logit;
            }
        }
    }

    // CHECKLIST: red_smem is used for a temporal buffer(shared memory) to compute block-wide sum
    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    // CHECKLIST: Compute the logits and start the sum.
    ////////////////////////////////////////////////////////////////////////////////////////////////

// Normalize the logits.
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float logit_scale = (FP8_KV_CACHE ? kv_scale_quant_orig_f : 1.0f);
#else
    float logit_scale = 1.f;
#endif // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float inv_sum = __fdividef(logit_scale, sum + 1.e-6f);

    const int normlization_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= normlization_loop_end; ti += THREADS_PER_BLOCK)
    {
        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        if (!MULTI_BLOCK_FLAG)
        {
            convert_from_float(&logits_smem[ti], qk_smem[ti] * inv_sum);
        }
        else
        {
            // no scaling factor inv_sum applied here, will apply the scaling factor after all blocks finished
            if (time_now < kv_loop_length && ti != timesteps_per_block)
            {
                convert_from_float(&logits_smem[ti], qk_smem[ti]);
            }
            else if (time_now == kv_loop_length)
            {
                convert_from_float(&logits_current_smem[0], qk_current_smem[0]);
            }
        }
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    const auto v_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
    // The value computed by this thread.
    const auto vo = v_idx.x;
    // The hidden dimensions computed by this particular thread.
    const auto vi = v_idx.y;
    // Base pointer for the row of pointers to v cache blocks
    void** v_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, batch_beam_idx));
    // Base pointer for the row of pointers to v cache blocks for beam's batch, before offsetting with indirection
    // buffer
    void** v_cache_batch_row_ptr
        = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, batch_idx * beam_width));

    // The number of values processed per iteration of the loop.
    constexpr unsigned V_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_VALUE};
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_V_PER_ITER = V_PER_ITER * V_LOOP_UNROLL;

    bool const is_valid_vi = IS_Dh_MAX || vi < Dh;

    // One group of threads computes the product(s) for the current timestep.
    V_vec_k v_bias;
    zero(v_bias);
    // if( vo == params.timestep % V_PER_ITER ) {
    if (is_valid_vi && HANDLE_KV && vo == kv_loop_length % V_PER_ITER)
    {
        // Trigger the loads from the V bias buffer.
        if (params.v_bias != nullptr)
        {
            const auto v_bias_offset = tensorrt_llm::common::flat_index2(hi_kv, vi, Dh);
            v_bias = *reinterpret_cast<const V_vec_k*>(&params.v_bias[v_bias_offset]);
        }

        if (DO_CROSS_ATTENTION)
        {
            *reinterpret_cast<V_vec_k*>(&bias_smem[vi]) = v_bias;
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // NOTE: Computing Weighted Sum of Values (V)
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
    using V_vec_accum = typename V_vec_accum_fp32_<V_vec_k>::Type;
#else
    using V_vec_accum = V_vec_k;
#endif
    // The partial outputs computed by each thread.
    V_vec_accum out;
    zero(out);

    // Loop over the timesteps to compute the partial outputs.
    if (is_valid_vi)
    {
        // Handle only context value cache with beam searching.
        // Handle both context and generation value cache without beam searching.
        // Explicit batching of LDGs (by V_LOOP_UNROLL) as it doesn't depend on indirection tables.
        // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
        const int context_length
            = DO_CROSS_ATTENTION ? kv_loop_length : (HAS_BEAMS ? beam0_context_length : kv_loop_length);
        int context_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : context_length;
        int generation_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
        for (int ti = vo; ti < context_v_loop_end; ti += UNROLLED_V_PER_ITER)
        {
            V_vec_m v_vec_cache[V_LOOP_UNROLL];
#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                // Fetch offset based on cache_indir when beam sampling
                int time_idx = ti + v_loop * V_PER_ITER + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                time_idx = min(time_idx, kv_loop_length - 1);
                int rowIdx = batch_idx * beam_width;

                const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                // The base pointer for the value in the cache buffer.
                Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                v_vec_cache[v_loop] = *reinterpret_cast<const V_vec_m*>(&v_cache_batch[inBlockIdx]);
            }

#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                V_vec_m v_vec = reinterpret_cast<V_vec_m*>(&v_vec_cache[v_loop])[0];

                int local_time_idx = ti + v_loop * V_PER_ITER;
                int time_idx = local_time_idx + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);

                const bool is_mask
                    = (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= context_length);

                // Load the logits from shared memory.
                // Note that fma will convert 8bit vec to the accumulation data type (float by default).
                Logit_value_fma<Tk, V_vec_accum, V_vec_m, INT8_KV_CACHE, FP8_KV_CACHE>(
                    out, reinterpret_cast<Tk*>(logits_smem + local_time_idx), v_vec, kv_scale_quant_orig_f, is_mask);
            }
        }

        // Handle generation value cache with beam searching.
        if (HAS_BEAMS && !DO_CROSS_ATTENTION)
        {
            const auto generation_start_ti
                = MULTI_BLOCK_FLAG ? vo : (vo + (beam0_context_length / V_PER_ITER) * V_PER_ITER);
            // Only the last few blocks need to handle the generation value cache.
            if (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > beam0_context_length)
            {
                for (int ti = generation_start_ti; ti < generation_v_loop_end; ti += V_PER_ITER)
                {
                    // Fetch offset based on cache_indir when beam sampling
                    int time_idx = ti + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                    int local_time_idx = ti;
                    if (time_idx < beam0_context_length || (MULTI_BLOCK_FLAG && time_idx >= kv_loop_length))
                    {
                        continue;
                    }
                    int rowIdx = batch_idx * beam_width + beam_indices[time_idx];

                    const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                    // The base pointer for the value in the cache buffer.
                    Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));
                    V_vec_m v_vec = reinterpret_cast<const V_vec_m*>(&v_cache_batch[inBlockIdx])[0];

                    // Load the logits from shared memory.
                    // Note that fma will convert 8bit vec to the accumulation data type (float by default).
                    Logit_value_fma<Tk, V_vec_accum, V_vec_m, INT8_KV_CACHE, FP8_KV_CACHE>(
                        out, reinterpret_cast<Tk*>(logits_smem + local_time_idx), v_vec, kv_scale_quant_orig_f, false);
                }
            }
        }
    }

    // Make sure we can overwrite the v cache if using cyclic kv cache.
    __syncthreads();

    // Get the c_tile_id that handles the current timestep.
    const int ctile_idx = tlength / timesteps_per_block;

    // One group of threads computes the product(s) for the current timestep.
    if (vo == kv_loop_length % V_PER_ITER && is_valid_vi && (!MULTI_BLOCK_FLAG || (c_tile == ctile_idx)))
    {
        const int tokenIdx = cyclic_tlength;
        const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tokenIdx, hi_kv, Dh, vi);
        // The base pointer for the value in the cache buffer.
        Tcache* v_cache_base = reinterpret_cast<Tcache*>(kvCacheBuffer.getBlockPtr(v_cache_base_row_ptr, tokenIdx));

        V_vec_k v;
        if (DO_CROSS_ATTENTION)
        {
            v = vec_conversion<V_vec_k, V_vec_k>(*reinterpret_cast<const V_vec_k*>(&v_cache_base[inBlockIdx]));
        }
        else
        {
            // Trigger the loads from the V buffer.
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            const auto v_offset = tensorrt_llm::common::flat_index_strided3(batch_beam_idx, hi_kv, vi, v_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_k>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec_k>::value>::type;
                const auto v_scaling = params.qkv_scale_quant_orig[2];
                const auto v_quant
                    = *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.v)[v_offset]);

                convert_from_float(&v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else
            {
                v = *reinterpret_cast<const V_vec_k*>(&params.v[v_offset]);
            }
        }

        if (HANDLE_KV)
        {
            // Compute the V values with bias.
            v = add(v, v_bias);

            if (do_ia3)
            {
                v = mul<V_vec_k, V_vec_k, V_vec_k>(v,
                    *reinterpret_cast<const V_vec_k*>(
                        &params.ia3_value_weights[tensorrt_llm::common::flat_index2(ia3_ti_hi, vi, Dh)]));
            }
        }

        // Store the values with bias back to global memory in the cache for V.
        //*reinterpret_cast<V_vec_k*>(&v_cache[params.timestep*Dh]) = v;
        // For MQA/GQA mode, write only with the first Q head of each group per KV head.
        if (hi == (hi_kv * qhead_per_kv))
        {
            if (ENABLE_8BITS_CACHE)
            {
                store_8bits_kv_cache_vec(v_cache_base, v, inBlockIdx, kv_scale_orig_quant);
            }
            else
            {
                *reinterpret_cast<V_vec_k*>(&v_cache_base[inBlockIdx]) = v;
            }
        }

        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)
        // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[kv_loop_length], cast_to_float(v), out);
        }
        else
        {
            out = fma(logits_current_smem[0], cast_to_float(v), out);
        }
#else  // MMHA_USE_FP32_ACCUM_FOR_LOGITS
       // out = fma(logits_smem[params.timestep], v, out);
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[kv_loop_length], v, out);
        }
        else
        { // MULTI_BLOCK_FLAG // Not supported yet: multi-block mode with FP8_MHA
            out = fma(logits_current_smem[0], v, out);
        }
#endif // MMHA_USE_FP32_ACCUM_FOR_LOGITS
    }
    // Make sure we can start writing to shared memory.
    __syncthreads();

    // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2)
    {

        // The midpoint in the number of active groups.
        int midpoint = active_groups / 2;

        // The upper part of active threads store to shared memory.
        if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh))
        {
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
            convert_from_float(reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
            *reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
        }
        __syncthreads();

        // The bottom warps update their values.
        if (vo < midpoint && (Dh == Dh_MAX || vi < Dh))
        {
            out = add(*reinterpret_cast<const V_vec_k*>(&out_smem[vo * Dh + vi]), out);
        }
        __syncthreads();
    }

    const auto bhi = tensorrt_llm::common::flat_index2(batch_beam_idx, hi, num_heads);
    const auto bhi_seq_len_tile = bhi * params.seq_len_tile;
    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh))
    {
        const auto bhvi = tensorrt_llm::common::flat_index2(bhi, vi, Dh);
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
        if (write_attention_quant)
        {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_accum>::value>::type;
            out = mul<V_vec_accum, float>(*params.attention_out_scale_orig_quant, out);
            *reinterpret_cast<Packed_Int8_t*>(&(reinterpret_cast<int8_t*>(params.out)[bhvi])) = cast_to_int8(out);
        }
        else
        {
            if (!MULTI_BLOCK_FLAG)
            {
                // This makes sure we have coalesced memory access.
                V_vec_k final_out;
                convert_from_float(&final_out, out);
                *reinterpret_cast<V_vec_k*>(&params.out[bhvi]) = final_out;
            }
            else
            {
                // for write partial output to partial_out
                int partial_out_offset = c_tile * params.batch_size * num_heads * params.hidden_size_per_head;
                // for write partial statistics to partial_max and partial_sum
                int partial_stats_offset = bhi_seq_len_tile + c_tile;

                // This makes sure we have coalesced memory access.
                V_vec_k partial_out;
                convert_from_float(&partial_out, out);
                *reinterpret_cast<V_vec_k*>(&params.partial_out[partial_out_offset + bhvi]) = partial_out;
                convert_from_float(reinterpret_cast<float*>(&params.partial_max[partial_stats_offset]), qk_max);
                convert_from_float(reinterpret_cast<float*>(&params.partial_sum[partial_stats_offset]), sum);
            }
        }
#else  // MMHA_USE_FP32_ACCUM_FOR_OUT
        *reinterpret_cast<V_vec_accum*>(&params.out[bhvi]) = out;
#endif // MMHA_USE_FP32_ACCUM_FOR_OUT
    }

#ifdef ENABLE_MULTI_BLOCK_OPTION
    if (MULTI_BLOCK_FLAG)
    {

        cuda::atomic_ref<int, cuda::thread_scope_device> count_ref{params.block_counter[bhi]};
        bool last_block{false};
        if (tidx == 0)
        {
            if (count_ref.fetch_add(1, cuda::memory_order_acq_rel) == (gridDim.z - 1))
            {
                last_block = true;
            }
        }

        ////////////////////
        ////////////////////
        // Make sure every threadblock finishes the previous computation, and enter the last threadblock in the
        // following (for each B and H) Do the final computation in the last threadblock Final reduction computation
        // by combining all the partial max/sum and outputs
        ////////////////////
        ////////////////////
        if (__syncthreads_or(last_block))
        {

            ////////////////////
            // Find the global max from all partial max -> use CUB BlockReduce
            ////////////////////

            float final_max = -FLT_MAX;
            float thread_partial_max = -FLT_MAX;
            thread_partial_max = params.partial_max[bhi_seq_len_tile + min(tidx, gridDim.z - 1)];

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // Specialize BlockReduce for a 1D block of THREADS_PER_BLOCK threads of type int
            typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
            // Allocate shared memory for BlockReduce
            __shared__ typename BlockReduce::TempStorage temp_storage;
            // Obtain a segment of consecutive items that are blocked across threads (final_max from above)
            // Compute the block-wide max for thread0
            final_max = BlockReduce(temp_storage).Reduce(thread_partial_max, cub::Max(), gridDim.z);

            __shared__ float final_max_smem;
            if (tidx == 0)
            {
                final_max_smem = final_max;
            }
            __syncthreads();

            // Finish the final_max computation
            final_max = final_max_smem;

            ////////////////////
            // Reduction for global sum over all partial sum (scaled by the exponential term from global max) -> use
            // gridDim.z threads
            ////////////////////

            float final_sum = 0.f;
            if (tidx < gridDim.z)
            {
                thread_partial_max = params.partial_max[bhi_seq_len_tile + tidx];
                const auto thread_partial_sum = params.partial_sum[bhi_seq_len_tile + tidx];
                final_sum += __expf(thread_partial_max - final_max) * thread_partial_sum;
            }

            // Compute the final_sum.
            final_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], final_sum);

            ////////////////////
            // Reduction for final output (scaled by the exponential term from global max) -> use THREADS_PER_VALUE
            // * gridDim.z threads
            ////////////////////

            // Shared memory to store partial outputs for each oi. -> size: gridDim.z * Dh * 4 Bytes. Reuse qk_smem.
            T* out_oi_smem = reinterpret_cast<T*>(smem_);

            const auto o_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);

            // Init partial out for accumulation.
            V_vec_k zero_k;
            zero(zero_k);
            V_vec_k thread_accumulated_out = zero_k;

            // The hidden dimensions computed by this particular thread. (refer to vi)
            const auto oi = o_idx.y;

            // The partial output region this thread takes care of
            const auto oo = o_idx.x;

            // Each thread may handle more than one partial output.
            for (int tile_idx = o_idx.x; tile_idx < gridDim.z; tile_idx += V_PER_ITER)
            {
                // Load partial output
                int thread_partial_out_offset = tile_idx * params.batch_size * num_heads * params.hidden_size_per_head;
                // Load partial max (different to thread_partial_max since the threadIdx rule changes here)
                float thread_partial_max_for_out = params.partial_max[bhi_seq_len_tile + tile_idx];
                // Load the partial outputs.
                V_vec_k thread_partial_out
                    = *reinterpret_cast<const V_vec_k*>(&params.partial_out[thread_partial_out_offset + bhi * Dh + oi]);
                // Apply the correction factor.
                Tk factor_compute;
                convert_from_float(&factor_compute, __expf(thread_partial_max_for_out - final_max));
                thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(factor_compute, thread_partial_out);
                thread_accumulated_out = add(thread_partial_out, thread_accumulated_out);
            }

            // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
            for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2)
            {

                // The midpoint in the number of active groups.
                int midpoint = active_groups / 2;

                // The upper part of active threads store to shared memory.
                if (oo >= midpoint && oo < active_groups && (Dh == Dh_MAX || oi < Dh))
                {
                    *reinterpret_cast<V_vec_k*>(&out_oi_smem[(oo - midpoint) * Dh + oi]) = thread_accumulated_out;
                }
                __syncthreads();

                // The bottom warps update their values.
                if (oo < midpoint && (Dh == Dh_MAX || oi < Dh))
                {
                    thread_accumulated_out
                        = add(thread_accumulated_out, *reinterpret_cast<const V_vec_k*>(&out_oi_smem[oo * Dh + oi]));
                }
                __syncthreads();
            }

            ////////////////////
            // Final output O * inv_sum
            ////////////////////

            if (oo == 0 && (Dh == Dh_MAX || oi < Dh))
            {
                const auto inv_sum = __fdividef(1.f, final_sum + 1.e-6f);

                Tk inv_sum_compute;
                convert_from_float(&inv_sum_compute, inv_sum);

                thread_accumulated_out = mul<V_vec_k, Tk, V_vec_k>(inv_sum_compute, thread_accumulated_out);
                *reinterpret_cast<V_vec_k*>(&params.out[bhi * Dh + oi]) = thread_accumulated_out;
            }

            // Reset qk_current_smem and block_counter for the next timestep
            if (tidx == 0)
            {
                params.block_counter[bhi] = 0;
            }
        }
    }
#endif // ENABLE_MULTI_BLOCK_OPTION
}

// __syncthreads();
// if (hi == 0 && tidx == 0)
// {
// #pragma unroll
//     for()
// }

} // namespace mmha

} // namespace kernels
} // namespace tensorrt_llm
