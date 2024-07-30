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

// [ ] Maybe
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h"

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

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

template <
    // The type of the inputs. Supported types: float, uint16_t, nv_bfloat16.
    typename T,
    // The type of the cache.
    typename Tcache,
    // The type of the shift key cache.
    typename TKcache,
    // Type of struct containing KV cache
    typename KVCacheBuffer,
    // Type of struct containing K cache to read past keys
    typename KCacheBuffer,
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
    // Whether enable position shift for streamingllm
    bool POS_SHIFT = false,
    // Whether to compute and apply block sparse attention mask
    bool BLOCK_SPARSE_ATTN = false,
    // Whether compute implicit relative attention bias on the fly.
    bool IMPLICIT_REL_ATTN_BIAS = false,
    // Whether apply tanh scale to the qk product.
    bool QK_TANH_SCALE = false,
    // The number of threads per key.
    unsigned THREADS_PER_KEY = threads_per_key<T, dh_max(Dh)>(),
    // The number of threads per value.
    unsigned THREADS_PER_VALUE = threads_per_value<T>(dh_max(Dh)),
    // The unroll factor for loading from K cache.
    // Set it default to 4 for higher occupancy (by reducing registers usage).
    unsigned K_LOOP_UNROLL = 4,
    // The unroll factor for loading from V cache.
    unsigned V_LOOP_UNROLL = 8,
    // Launch bounds
    unsigned MAX_THEADS_PER_BLOCK
    = Launch_bounds_config<T, Tcache, THREADS_PER_BLOCK, dh_max(Dh), DO_CROSS_ATTENTION, HAS_BEAMS, POS_SHIFT>()
          .MAX_THREADS_PER_BLOCK,
    unsigned MIN_BLOCKS_PER_SM
    = Launch_bounds_config<T, Tcache, THREADS_PER_BLOCK, dh_max(Dh), DO_CROSS_ATTENTION, HAS_BEAMS, POS_SHIFT>()
          .MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(MAX_THEADS_PER_BLOCK, MIN_BLOCKS_PER_SM) masked_multihead_attention_kernel_1(
    Multihead_attention_params<T, DO_CROSS_ATTENTION> params, KVCacheBuffer kvCacheBuffer, KCacheBuffer pastKCache)
{

    using Tk = typename kernel_type_t<T>::Type;
    // Use 8bit cache.
    static constexpr bool ENABLE_8BITS_K_CACHE = sizeof(TKcache) == 1;
    static constexpr bool ENABLE_8BITS_KV_CACHE = sizeof(Tcache) == 1;
    // FP8 KV Cache.
    static constexpr bool FP8_K_CACHE = std::is_same<TKcache, __nv_fp8_e4m3>::value;

    // Never Used
    // static constexpr bool FP8_KV_CACHE = std::is_same<Tcache, __nv_fp8_e4m3>::value;

    // INT8 KV Cache.
    // Never Used
    // static constexpr bool INT8_KV_CACHE = std::is_same<Tcache, int8_t>::value;

    // The size of a warp.
    constexpr unsigned WARP_SIZE{32};
    // The number of warps in a threadblock.
    constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

    // The maximum hidden size per head.
    constexpr auto Dh_MAX = dh_max(Dh);
    // Never Used
    // constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
    static_assert(Dh_MAX >= WARP_SIZE);
    static_assert(Dh_MAX >= Dh);
    // Only instantiate few head sizes for implicit relative attention bias in order to save compilation time.
    static_assert(!IMPLICIT_REL_ATTN_BIAS || Dh == 32 || Dh == 64 || Dh == 128);

    // The maximum sequence length in the cyclic kv_cache, i.e., an upper bound on L.
    // Note that the maximum sequence length supported by the model might be greater than this.
    // Note max_attention_window_size is maximum of cyclic_attention_window_size among all layers.
    // By default, you can assume that they are the same.
    auto const cyclic_kv_cache_len = static_cast<unsigned>(params.cyclic_attention_window_size);
    // The number of sink tokens in kv cache to support streamingllm
    auto const sink_token_len = static_cast<unsigned>(params.sink_token_length);
    // The current timestep (including paddings).
    // It is only used to calculate the smem stride.
    auto const timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep);

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
        auto const max_timesteps = DO_CROSS_ATTENTION ? cyclic_kv_cache_len : min(timestep, cyclic_kv_cache_len);
        logits_smem_ += divUp(max_timesteps + 1, 4u) * 16;
    }
    // Never Used
    // Tk* logits_smem = reinterpret_cast<Tk*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    __shared__ Tk logits_current_smem[1];

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    // Never Used
    // Tk* out_smem = reinterpret_cast<Tk*>(smem_);

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
    using K_vec_m = typename packed_type<TKcache, num_elems<K_vec_k>::value>::type;
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
        [[maybe_unused]] Tk bias_smem[bias_smem_size];

    // The number of elements per vector.
    constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0);
    // We will use block wide reduction if needed
    // The number of vectors per Dh_MAX.
    constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
    static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

    // The batch/beam idx
    auto const batch_beam_idx = blockIdx.y;
    if (params.finished != nullptr && params.finished[batch_beam_idx])
    {
        return;
    }

    // The head.
    unsigned const hi{blockIdx.x};
    // The head index of keys and values adjusted for MQA/GQA.
    int const qhead_per_kv{params.num_heads / params.num_kv_heads};
    unsigned const hi_kv{hi / qhead_per_kv};
    // The number of heads.
    auto const num_heads = static_cast<unsigned>(params.num_heads);
    // The number of heads for keys and values adjusted for MQA/GQA.
    auto const num_heads_kv = static_cast<unsigned>(params.num_kv_heads);

    // The thread in the block.
    unsigned const tidx{threadIdx.x};

    // The column tile along L dimension on K^T -- noted as T_c in flash-attention paper
    unsigned const c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};

    // Indicate if we need to compute the K/V cache element (add KV bias, IA3, RoPE, etc.) and update the cache.
    // For Self-Attention, it's always required.
    // For Cross-Attention, as everything is pre-computed,
    // in the context phase of the encoder, it's not needed in that kernel.
    // Therefore, HANDLE_KV is !DO_CROSS_ATTENTION and irrelevant of timestep.
    static constexpr bool HANDLE_KV{!DO_CROSS_ATTENTION};

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    // Do we have a relative attention bias?
    bool has_relative_attention_bias = params.relative_attention_bias != nullptr;
    // IMPLICIT_REL_ATTN_BIAS:
    // Compute relative attention bias on the fly, with relative attention table [head_num/TP, num_buckets] passed in.
    // num_buckets passed as relative_attention_bias_stride, max_distance passed as params.max_distance
    // this is a common optimization for both self attention and cross attention
    int relative_attention_bias_stride
        = params.relative_attention_bias_stride; // num_buckets might be modified below, save it beforehand
    [[maybe_unused]] int max_distance = params.max_distance;

    // The actual sequence length excluding the paddings.
    // minus 1 because it includes the current timestep while tlength denotes the kv cache length.
    int const tlength = DO_CROSS_ATTENTION
        ? params.memory_length_per_sample[batch_beam_idx] - 1
        : (params.length_per_sample ? (params.length_per_sample[batch_beam_idx] - 1) : static_cast<int>(timestep));
    // We will use cyclic kv cache when it exceeds the limit.
    // The length position for storing new key and value.
    int const cyclic_tlength = kvCacheBuffer.getKVTokenIdx(tlength);
    // When enable cyclic kv cache and one more block mode, we need to shift the index to the actual index in the
    // sequence. Otherwise, if the token is not the sink token, we need to add the bubblen length to the index.
    bool const enable_use_seq_idx_kv = kvCacheBuffer.mEnableOneMoreBlock && tlength > cyclic_kv_cache_len;
    int const shift_for_cyclic_kv = (enable_use_seq_idx_kv) ? tlength - cyclic_kv_cache_len : kvCacheBuffer.mBubbleLen;
    int const shift_for_cyclic_k = (enable_use_seq_idx_kv) ? tlength - cyclic_kv_cache_len : pastKCache.mBubbleLen;
    // The actual kv cache length.
    // tlength is the past length actually.
    int const kv_loop_length = min(tlength, cyclic_kv_cache_len);
    // The context length for beam searching optimization (all points to beam 0).
    // TODO: with cyclic kv cache, we set it 0 for now (will optimize in the future)
    // as context kv cache might be overwritten by the new kv cache
    int const beam0_context_length
        = HAS_BEAMS && tlength > cyclic_kv_cache_len ? 0 : params.input_lengths[batch_beam_idx];
    // The position of the current timestep, and it is used to apply the position embedding
    int const current_pos_idx = (!POS_SHIFT || DO_CROSS_ATTENTION) ? tlength : kv_loop_length;

    // The offset in the Q and K buffer also accounts for the batch.
    auto const qk_vec_idx = tidx * QK_VEC_SIZE;
    auto const is_valid_qk_vec = qk_vec_idx < Dh;

    // [x] Store qk_values
    auto const max_attention_window_size = params.max_attention_window_size;
    int const qk_values_offset = hi * max_attention_window_size;

    bool const load_qkv_quant = params.qkv_scale_quant_orig != nullptr;
    bool const write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

    // Quant/Dequant scales for 8bits kv cache.
    using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
    T_scale kv_scale_orig_quant, k_scale_quant_orig;
    float const k_scale_quant_orig_f = (ENABLE_8BITS_K_CACHE ? params.kv_scale_quant_orig[0] : 1.0f);
    float const kv_scale_quant_orig_f = (ENABLE_8BITS_KV_CACHE ? params.kv_scale_quant_orig[0] : 1.0f);
    convert_from_float(&k_scale_quant_orig, k_scale_quant_orig_f);
    convert_from_float(&kv_scale_orig_quant, (ENABLE_8BITS_KV_CACHE ? params.kv_scale_orig_quant[0] : 1.0f));

    // Up to QK_VECS_PER_Dh_MAX threads load Q and K + the bias values for the current timestep.
    // Trigger the loads from the Q and K buffers.
    Qk_vec_k q, k, q_bias, k_bias;
    // key without position embedding
    Qk_vec_k k_wo_pos;
    zero(q);
    zero(k);
    zero(q_bias);
    zero(k_bias);
    zero(k_wo_pos);
    float rotary_embedding_base = params.rotary_embedding_base;
    float rotary_embedding_scale = params.rotary_embedding_scale;
    if (is_valid_qk_vec)
    {
        mmha::update_rotary_base_n_scale(rotary_embedding_base, rotary_embedding_scale,
            params.rotary_embedding_scale_type, params.rotary_embedding_dim, params.rotary_embedding_max_positions,
            current_pos_idx);
        // Query
        // The stride between tokens. We may be able to always use params.stride.
        uint32_t q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * Dh);
        // The offset.
        auto const q_offset = tensorrt_llm::common::flat_index_strided3(batch_beam_idx, hi, qk_vec_idx, q_stride, Dh);

        if (load_qkv_quant)
        {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
            auto const q_scaling = params.qkv_scale_quant_orig[0];
            auto const q_quant
                = *reinterpret_cast<Packed_Int8_t const*>(&reinterpret_cast<int8_t const*>(params.q)[q_offset]);
            convert_from_float(&q, mul<Packed_Float_t, float>(q_scaling, float_from_int8(q_quant)));
        }
        else
        {
            q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.q[q_offset]));
        }

        if constexpr (DO_CROSS_ATTENTION)
        {
            auto const k_idx = QK_VEC_SIZE * tidx;
            int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(cyclic_tlength, hi, Dh, k_idx);
            Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, cyclic_tlength));

            k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&k_cache[inBlockIdx]));
        }
        else
        {
            // Key
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t k_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            auto const k_offset
                = tensorrt_llm::common::flat_index_strided3(batch_beam_idx, hi_kv, qk_vec_idx, k_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
                auto const k_scaling = params.qkv_scale_quant_orig[1];
                auto const k_quant
                    = *reinterpret_cast<Packed_Int8_t const*>(&reinterpret_cast<int8_t const*>(params.k)[k_offset]);

                convert_from_float(&k, mul<Packed_Float_t, float>(k_scaling, float_from_int8(k_quant)));
            }
            else
            {
                k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.k[k_offset]));
            }
        }

        if (params.q_bias != nullptr)
        {
            auto const q_bias_offset = tensorrt_llm::common::flat_index2(hi, qk_vec_idx, Dh);
            q_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.q_bias[q_bias_offset]));
        }
        if (HANDLE_KV && params.k_bias != nullptr)
        {
            auto const k_bias_offset = tensorrt_llm::common::flat_index2(hi_kv, qk_vec_idx, Dh);
            k_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.k_bias[k_bias_offset]));
        }
    }

    // Computes the Q/K values with bias.
    q = add(q, q_bias);
    if (HANDLE_KV)
    {
        k = add(k, k_bias);
    }

    // The width of the beam.
    auto const beam_width = static_cast<unsigned>(params.beam_width);
    // The batch idx.
    int const batch_idx = batch_beam_idx / beam_width;
    // Do we apply IA3?
    bool const do_ia3 = HANDLE_KV && params.ia3_tasks != nullptr;
    // Compute the IA3 task. One per batch index.
    auto const ia3_ti_hi = do_ia3
        ? tensorrt_llm::common::flat_index2(static_cast<unsigned>(params.ia3_tasks[batch_idx]), hi, num_heads)
        : 0;

    if (do_ia3 && is_valid_qk_vec)
    {
        k = mul<Qk_vec_k, Qk_vec_k, Qk_vec_k>(k,
            vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(
                &params.ia3_key_weights[tensorrt_llm::common::flat_index2(ia3_ti_hi, qk_vec_idx, Dh)])));
    }
    k_wo_pos = k;

    // Note we have no paddings in KV cache now.
    switch (params.position_embedding_type)
    {
    case PositionEmbeddingType::kLEARNED_ABSOLUTE:
    case PositionEmbeddingType::kRELATIVE:
    case PositionEmbeddingType::kALIBI:
    case PositionEmbeddingType::kALIBI_WITH_SCALE:
    {
        break;
    }
    case PositionEmbeddingType::kROPE_GPTJ:
    {
        if (HANDLE_KV)
        {
            apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, rotary_embedding_base,
                rotary_embedding_scale, 0, nullptr, current_pos_idx);
        }
        else
        {
            apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale,
                0, nullptr, current_pos_idx);
        }
        break;
    }
    case PositionEmbeddingType::kLONG_ROPE:
    case PositionEmbeddingType::kROPE_GPT_NEOX:
    {
        bool const do_rotary = is_valid_qk_vec && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

        T* q_smem_ = reinterpret_cast<T*>(smem_);
        T* k_smem_ = q_smem_ + params.rotary_embedding_dim;

        int const half_rotary_dim = params.rotary_embedding_dim / 2;
        int const half_idx = qk_vec_idx / half_rotary_dim;
        int const intra_half_idx = qk_vec_idx % half_rotary_dim;
        int const smem_pitch = half_rotary_dim; // TODO: adjust for bank conflicts

        assert(half_rotary_dim % QK_VEC_SIZE == 0);

        if (do_rotary)
        {
            *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx) = q;
            if (HANDLE_KV)
            {
                *reinterpret_cast<Qk_vec_k*>(k_smem_ + half_idx * smem_pitch + intra_half_idx) = k;
            }
        }

        __syncthreads();

        int const transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
        if (do_rotary)
        {
            float rotary_embedding_m_scale = tlength <= params.rotary_embedding_original_max_positions
                ? params.rotary_embedding_short_m_scale
                : params.rotary_embedding_long_m_scale;
            mmha::vec_from_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
            if (HANDLE_KV)
            {
                mmha::vec_from_smem_transpose(k, k_smem_, transpose_idx, smem_pitch);

                mmha::apply_rotary_embedding(q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, rotary_embedding_m_scale,
                    params.rotary_embedding_scaling_factors, current_pos_idx, params.rotary_cogvlm_vision_start,
                    params.rotary_cogvlm_vision_length);

                mmha::write_smem_transpose(k, k_smem_, transpose_idx, smem_pitch);
            }
            else
            {
                mmha::apply_rotary_embedding(q, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, rotary_embedding_m_scale,
                    params.rotary_embedding_scaling_factors, current_pos_idx, params.rotary_cogvlm_vision_start,
                    params.rotary_cogvlm_vision_length);
            }
            mmha::write_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary)
        {
            q = *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx);
            if (HANDLE_KV)
            {
                k = *reinterpret_cast<Qk_vec_k*>(k_smem_ + half_idx * smem_pitch + intra_half_idx);
            }
        }

        __syncthreads();
        break;
    }
    }

    // For the same reason as HANDLE_KV, no compute needed in Cross-Attention's 1st step
    // Store Q K vectors to shared memory, and calculate QK.
    if (qk_vec_idx < Dh_MAX)
    {

        // Store the Q values to shared memory.
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
        if constexpr (FP8_K_CACHE)
        {
            // There are many more elements from K than elements from Q so we pre-scale Q instead
            // of scaling all the elements from K. It helps reduce the number of ops.
            Qk_vec_k scaled_q;
            zero(scaled_q);
            if (is_valid_qk_vec)
            {
                scaled_q = mul<Qk_vec_k, Tk, Qk_vec_k>(k_scale_quant_orig, q);
            }
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = scaled_q;
        }
        else
#endif
        {
            // Set padded Dh to 0 for the correctness of QK (when Dh != Dh_Max).
            Qk_vec_k zero_q;
            zero(zero_q);
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = is_valid_qk_vec ? q : zero_q;
        }

        // Store the K values to shared memory.
        // We store K values from shared memory to global memory
        //  when the target position of K cache in global memory has been accessed (in the case of cyclic kv cache)
        if (POS_SHIFT && !DO_CROSS_ATTENTION)
        {
            reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx])[0] = k_wo_pos;
        }
        else
        {
            reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx])[0] = k;
        }

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
        qk = dot<Qk_vec_accum, Qk_vec_k>(q, k);
        if (QK_VECS_PER_Dh_MAX <= WARP_SIZE)
        {
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
            {
                qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_Dh_MAX > WARP_SIZE)
    {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_Dh_MAX + WARP_SIZE - 1) / WARP_SIZE;
        qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Pre-compute the pointer for the relative attention bias.
    T const* relative_attention_bias_ptr = nullptr;
    [[maybe_unused]] T const* relative_attention_bias_ptr_fixed = nullptr; // record the base for offset
    if (has_relative_attention_bias)
    {
        // "hi" is unsigned, subtracting int from unsigned int causes underflow. Cast to int
        int64_t offset = IMPLICIT_REL_ATTN_BIAS
            ? ((int64_t) hi * relative_attention_bias_stride - tlength)
            : ((int64_t) hi * relative_attention_bias_stride + tlength) * relative_attention_bias_stride;
        relative_attention_bias_ptr = &params.relative_attention_bias[offset];
        relative_attention_bias_ptr_fixed = &params.relative_attention_bias[offset];
    }

    // Load the value.
    float relative_attention_bias = 0.f;
    if (has_relative_attention_bias && tidx == 0)
    {
        // TODO: Use a better way to convert from T to float.
        relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[tlength]);
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0)
    {
        // Normalize qk.
        qk = qk * params.inv_sqrt_dh + relative_attention_bias;

        // Grok tanh scale for qk product.
        if constexpr (QK_TANH_SCALE)
        {
            qk = params.qk_tanh_scale * tanhf(qk * params.qk_tanh_inverse_scale);
        }

        // We don't need to apply the linear position bias here since qi - ki = 0 yields the position bias 0.
        qk_max = qk;

        // Store Q*K^T to shared memory.
        if (MULTI_BLOCK_FLAG)
        {
            qk_current_smem[0] = qk;
        }
        else
        {
            // We need to store the qk result to the end of the qk_smem for cyclic kv cache (+ 1 for smem memory
            // allocation) because the previous cache will still write to the new_cache_pos of qk_smem.
            qk_smem[kv_loop_length] = qk;
        }
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

    // The positions of the cache buffer (for this B * H) and the vector within that chunk associated with this
    // thread.
    auto const k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

    // The number of vectors per thread.
    constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
    static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec_accum q_vec[K_VECS_PER_THREAD];
#pragma unroll
    for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
    {
        q_vec[ii] = vec_conversion<K_vec_accum, K_vec_k>(*reinterpret_cast<K_vec_k const*>(
            &q_smem[tensorrt_llm::common::flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]));
    }

    // The number of timesteps loaded per iteration, i.e., (THREADS_PER_BLOCK * THREADS_PER_BLOCK) / 256 <= 256
    constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
    // The number of keys per warp.
    constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
    // The number of unrolled keys per warp.
    constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

    auto const timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
    int const context_length
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

    auto const context_ti_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP
        : divUp(static_cast<unsigned>(context_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

    // The generation ti_end.
    auto const generation_ti_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, K_PER_WARP) * K_PER_WARP
        : divUp(static_cast<unsigned>(kv_loop_length), K_PER_WARP) * K_PER_WARP;

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    // Note max_attention_window_size is maximum of cyclic_attention_window_size among all layers.
    // By default, you can assume that they are the same.
    auto const bi_seq_len_offset = static_cast<std::size_t>(batch_beam_idx) * params.max_attention_window_size;
    // Beam indices are based on the max_attention_window_size while each layer may have different
    // cyclic_attention_window_size So we need to rebuild the beam_indices if max_attention_window_size is not equal to
    // cyclic_attention_window_size.
    int const* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    auto const c_tile_times_timesteps_per_block = c_tile * timesteps_per_block; // 0 if !MULTI_BLOCK_FLAG

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Key cache loops for dot(Q, K).

    // Is it the leader?
    bool const is_leader = Qk_dot<T, THREADS_PER_KEY>::is_leader(tidx);

    // The slope for ALiBi.
    float linear_bias_slope = 0.f;
    if (params.linear_bias_slopes != nullptr)
    {
        // TODO: Use a cleaner code to convert from T to float.
        linear_bias_slope = mul<float>(params.linear_bias_slopes[hi], 1.f);
    }

    // Handle only context key cache with beam searching.
    // Handle both context and generation key cache without beam searching.
    // Explicit batching of LDGs (by K_LOOP_UNROLL) as it doesn't depend on indirection tables.
    for (int ti = k_idx.x; ti < context_ti_end; ti += UNROLLED_K_PER_ITER)
    {
        int const time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        // The keys loaded from the key cache.
        K_vec_m k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
        {
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                // Make sure we read data within the bound.
                // Dh OOB values will be handled by zero_q.
                // Seq OOB values will be masked out when storing back to smem.
                auto const jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                int valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);
                if (POS_SHIFT && valid_time_now >= sink_token_len)
                {
                    // If one more block mode is enabled, we use the index in sequence as tokenIdx.
                    // Otherwise, we need to add the bubble length to the index
                    valid_time_now += shift_for_cyclic_k;
                    if (enable_use_seq_idx_kv)
                    {
                        // Convert the token index in sequence to token index in K cache.
                        valid_time_now = pastKCache.getKVTokenIdx(valid_time_now);
                    }
                }
                int const seqIdx = batch_idx * beam_width;

                // Base pointer to k cache block for beam's batch
                TKcache* k_cache_batch = reinterpret_cast<TKcache*>(pastKCache.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx = pastKCache.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                k_vec_cache[k_loop][k_vec_i] = *reinterpret_cast<K_vec_m const*>(&k_cache_batch[inBlockIdx]);
            }
        }

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
        {
            int const local_time_now = time_now + k_loop * K_PER_ITER;
            int const local_ti = ti + k_loop * K_PER_ITER;

            // Perform the dot product and normalize qk.
            //
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            K_vec_m k_vec[K_VECS_PER_THREAD];
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                k_vec[k_vec_i] = *reinterpret_cast<K_vec_m*>(&k_vec_cache[k_loop][k_vec_i]);
            }

            // Is it active?
            bool const is_active = local_time_now < context_length;

            if constexpr (IMPLICIT_REL_ATTN_BIAS)
            {
                // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                int relative_buckets = 0;
                int relative_position = local_time_now - tlength;
                int num_buckets = relative_attention_bias_stride;
                // Special logic in T5 relative attention, both encoder & decoder use this, because
                // relative_attention_bias is pre-computed once and passed around.
                // T5 decoder attention now only uses bidirectional=False relative position logic
                // (ref: tensorrt_llm/layers/attention.py compute_relative_bias())
                relative_position = relative_position >= 0 ? 0 : -relative_position;

                int max_exact = num_buckets / 2;
                bool is_small = relative_position < max_exact;
                int relative_position_if_large = max_exact
                    + (int) (logf(relative_position * 1.0f / max_exact) / logf((float) max_distance / max_exact)
                        * (num_buckets - max_exact));
                relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                relative_buckets += is_small ? relative_position : relative_position_if_large;
                relative_attention_bias_ptr
                    = relative_attention_bias_ptr_fixed + (tlength - local_time_now) + relative_buckets;
            }

            // Prefetch the relative attention bias.
            float relative_attention_bias = 0.f;
            if (is_active && has_relative_attention_bias)
            {
                // TODO: Use a better way to convert from T to float.
                relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[local_time_now]);
            }

            // Compute the dot product between Q and K.
            // Note that dot will convert 8bit vec to the accumulation data type (float by default).
            float qk_ = 0.f;
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            if constexpr (FP8_K_CACHE)
            {
                qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
            }
            else
#endif // MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            {
                if constexpr (ENABLE_8BITS_K_CACHE)
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::scale_dot(q_vec, k_vec, k_scale_quant_orig_f)
                        * params.inv_sqrt_dh;
                }
                else
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
                }
            }

            // Grok tanh scale for qk product.
            if constexpr (QK_TANH_SCALE)
            {
                qk_ = params.qk_tanh_scale * tanhf(qk_ * params.qk_tanh_inverse_scale);
            }

            // For multi-block mode, we need to make sure it will not be OOB.
            if (MULTI_BLOCK_FLAG && local_ti >= timesteps_per_block)
            {
                continue;
            }

            // Add the ALiBi bias. (ki - qi) * slope[hi].
            //
            // The padding tokens are located between the input context and the generated tokens.
            // We need to remove the correct number of padding tokens in the distance computation.
            //
            //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
            //   token: i i i i p p p o o o where i=input, p=pad, o=output.
            // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
            //
            // All the threads do the work even if it's not relevant to avoid divergence.
            qk_ += linear_bias_slope * (local_time_now - tlength) + relative_attention_bias;

            if constexpr (BLOCK_SPARSE_ATTN)
            {
                float mask_val
                    = params.block_sparse_params.computeMask(tlength, local_time_now, tlength + 1, num_heads, hi) ? 1.f
                                                                                                                  : 0.f;
                qk_ += (1.0f - mask_val) * -10000.0f;
            }

            // There's one qk value per timestep.
            // Make sure only leader threads stores qk value within the bound.
            if (is_active && is_leader)
            {
                // Calculate the max for softmax.
                qk_max = fmaxf(qk_max, qk_);
                // Store the product to shared memory.
                qk_smem[local_ti] = qk_;
            }
        }
    }

    // Handle generation key cache with beam searching.
    // Note that it may be overlapped with the context key loop, but it won't impact the corretness.
    // Can skip in cross attention mode.
    if (HAS_BEAMS && !DO_CROSS_ATTENTION
        && (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > beam0_context_length))
    {
        // The input length;
        int const input_length_ = MULTI_BLOCK_FLAG ? beam0_context_length % timesteps_per_block : beam0_context_length;
        // The beginning of the generation.
        int const generation_start_ti = k_idx.x + input_length_ / K_PER_WARP * K_PER_WARP;

        // Iterate over the output tokens.
        for (int ti = generation_start_ti; ti < generation_ti_end; ti += K_PER_ITER)
        {
            int const time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

            // The keys loaded from the key cache.
            K_vec_m k_vec[K_VECS_PER_THREAD];

#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                int const jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                int valid_time_now = min(time_now, kv_loop_length - 1);
                int beam_offset = beam_indices[valid_time_now];
                if (POS_SHIFT && valid_time_now >= sink_token_len)
                {
                    // If one more block mode is enabled, we use the index in sequence as tokenIdx.
                    // Otherwise, we need to add the bubble length to the index
                    valid_time_now += shift_for_cyclic_k;
                    if (enable_use_seq_idx_kv)
                    {
                        // Convert the token index in sequence to token index in K cache.
                        valid_time_now = pastKCache.getKVTokenIdx(valid_time_now);
                    }
                }
                int const seqIdx = batch_idx * beam_width + beam_offset;
                // Base pointer to k cache block for beam's batch, before offsetting with indirection buffer
                TKcache* k_cache_batch = reinterpret_cast<TKcache*>(pastKCache.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx = pastKCache.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                k_vec[k_vec_i] = (*reinterpret_cast<K_vec_m const*>(&k_cache_batch[inBlockIdx]));
            }

            // Is it active?
            bool const is_active = time_now >= context_length && time_now < kv_loop_length;

            if constexpr (IMPLICIT_REL_ATTN_BIAS)
            {
                // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                int relative_buckets = 0;
                int relative_position = time_now - tlength;
                int num_buckets = relative_attention_bias_stride;
                // Special logic in T5 relative attention, both encoder & decoder use this, because
                // relative_attention_bias is pre-computed once and passed around.
                // T5 decoder attention now only uses bidirectional=False relative position logic
                // (ref: tensorrt_llm/layers/attention.py compute_relative_bias())
                relative_position = relative_position >= 0 ? 0 : -relative_position;

                int max_exact = num_buckets / 2;
                bool is_small = relative_position < max_exact;
                int relative_position_if_large = max_exact
                    + (int) (logf(relative_position * 1.0f / max_exact) / logf((float) max_distance / max_exact)
                        * (num_buckets - max_exact));
                relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                relative_buckets += is_small ? relative_position : relative_position_if_large;
                relative_attention_bias_ptr
                    = relative_attention_bias_ptr_fixed + (tlength - time_now) + relative_buckets;
            }

            // Prefetch the relative attention bias.
            float relative_attention_bias = 0.f;
            if (is_active && has_relative_attention_bias)
            {
                // TODO: Use a better way to convert from T to float.
                relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[time_now]);
            }

            // Perform the dot product and normalize qk.
            //
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            // Note that dot will convert 8bit vec to the accumulation data type (float by default).
            float qk_ = 0.f;
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            if constexpr (FP8_K_CACHE)
            {
                qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
            }
            else
#endif // MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            {
                if constexpr (ENABLE_8BITS_K_CACHE)
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::scale_dot(q_vec, k_vec, k_scale_quant_orig_f)
                        * params.inv_sqrt_dh;
                }
                else
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
                }
            }

            // Grok tanh scale for qk product.
            if constexpr (QK_TANH_SCALE)
            {
                qk_ = params.qk_tanh_scale * tanhf(qk_ * params.qk_tanh_inverse_scale);
            }

            // Add the ALiBi bias. (ki - qi) * slope[hi].
            //
            // The padding tokens are located between the input context and the generated tokens.
            // We need to remove the correct number of padding tokens in the distance computation.
            //
            //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
            //   token: i i i i p p p o o o where i=input, p=pad, o=output.
            // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
            //
            // All the threads perform that step to avoid divergence.
            qk_ += linear_bias_slope * (time_now - tlength) + relative_attention_bias;

            if constexpr (BLOCK_SPARSE_ATTN)
            {
                float mask_val
                    = params.block_sparse_params.computeMask(tlength, time_now, tlength + 1, num_heads, hi) ? 1.f : 0.f;
                qk_ += (1.0f - mask_val) * -10000.0f;
            }

            // There's one qk value per timestep.
            // Make sure only leader threads stores qk value within the bound.
            if (is_active && is_leader)
            {
                // Calculate the max for softmax.
                qk_max = fmaxf(qk_max, qk_);
                // Store the product to shared memory.
                qk_smem[ti] = qk_;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Softmax.

    // Perform the final reduction to compute the max inside each warp.
    //
    // NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
    // group so it's not needed to run the reduction inside the group (again).

#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
    // Leader threads will be in the dignonal when using HMMA.
    if (THREADS_PER_KEY <= 4)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 4));
    }
    if (THREADS_PER_KEY <= 8)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 9));
    }
    if (THREADS_PER_KEY <= 16)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 18));
    }
#else
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
#endif // defined MMHA_USE_HMMA

    // Decompose the thread index into warp and lane.
    auto const warp = tidx / WARP_SIZE;
    auto const lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0)
    {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

    // After the syncthreads, the target k position (cyclic kv cache) should also have been used by the k loop.
    // Write the K values to the global memory cache.
    //
    // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
    // system. We designed it this way as it allows much better memory loads (and there are many
    // more loads) + the stores are really "write and forget" since we won't need the ack before
    // the end of the kernel. There's plenty of time for the transactions to complete.

    // For MQA/GQA mode, write only with the first Q head of each group per KV head.

    // Get the c_tile_id that handles the current timestep.
    int current_step_ctile_idx = kv_loop_length / timesteps_per_block;
    if (HANDLE_KV && hi == (hi_kv * qhead_per_kv) && qk_vec_idx < Dh
        && (!MULTI_BLOCK_FLAG || c_tile == current_step_ctile_idx))
    {
        // Trigger the stores to global memory.
        Qk_vec_k k_vec = *reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx]);
        auto const k_idx = QK_VEC_SIZE * tidx;
        int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(cyclic_tlength, hi_kv, Dh, k_idx);
        // The base pointer for the value in the cache buffer.
        Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, cyclic_tlength));

        if constexpr (ENABLE_8BITS_KV_CACHE)
        {
            store_8bits_kv_cache_vec(reinterpret_cast<Tcache*>(k_cache), k_vec, inBlockIdx, kv_scale_orig_quant);
        }
        else
        {
            *reinterpret_cast<Qk_vec_m*>(&k_cache[inBlockIdx]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k_vec);
        }
        // [ ] Store the key with full-precision into the kv_cache_full
    }

    // The warps finalize the reduction.
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

// CHECKLIST: Split-Point=============================================================================================
// [x] Store the qk_values and qk_max_values into the intermediate buffers
#pragma unroll
    for (int out_i = tidx; out_i <= kv_loop_length; out_i += THREADS_PER_BLOCK)
    {
        params.qk_values[qk_values_offset + out_i] = qk_smem[out_i];
    }
    // }
    if (tidx == 0)
    {
        params.qk_max_values[hi] = qk_max;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // TODO_: Perform Topk operation

    int const MAX_K = 10;

    float const MAX_T_VAL = 65504.F;
    TopK<float, MAX_K> partial;

    typedef cub::BlockReduce<TopK<float, MAX_K>, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Initilization
#pragma unroll
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

    // Insertion of values

#pragma unroll
    for (int i = tidx; i <= kv_loop_length; i += THREADS_PER_BLOCK)
    {
        // int index = bid * MAX_K * MAX_K + i * THREADS_PER_BLOCK + tid;
        partial.insert(qk_smem[i], i);
    }

    // Block-wide Reduction
    TopK<float, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<float, MAX_K>);

    // DEBUGGING
    if (hi == 0 && tidx == 0)
    {
        printf("Total qk values: ");
        for (int i = 0; i < 10; i += 1)
        {
            printf("%f, ", total.u[i]);
        }
        printf("\n");

        printf("qk values: ");
        for (int i = 0; i <= kv_loop_length; i++)
        {
            printf("%f, ", qk_smem[i]);
        }
        printf("\n");
    }

    // Store the topk value for each head into the global memory
    int const topk_qk_index = hi * MAX_K;
#pragma unroll
    for (int i = 0; i < MAX_K; i += THREADS_PER_BLOCK)
    {
        params.topk_qk_indices[topk_qk_index + i] = total.p[i];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // TODO_: Transfer some data to Host
}

} // namespace mmha

} // namespace kernels
} // namespace tensorrt_llm
