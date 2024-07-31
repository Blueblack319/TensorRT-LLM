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
__global__ void __launch_bounds__(MAX_THEADS_PER_BLOCK, MIN_BLOCKS_PER_SM) masked_multihead_attention_kernel_2(
    Multihead_attention_params<T, DO_CROSS_ATTENTION> params, KVCacheBuffer kvCacheBuffer, KCacheBuffer pastKCache)
{

    using Tk = typename kernel_type_t<T>::Type;
    // Use 8bit cache.
    static constexpr bool ENABLE_8BITS_K_CACHE = sizeof(TKcache) == 1;
    static constexpr bool ENABLE_8BITS_KV_CACHE = sizeof(Tcache) == 1;
    // FP8 KV Cache.
    static constexpr bool FP8_K_CACHE = std::is_same<TKcache, __nv_fp8_e4m3>::value;
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
    // Never used
    // constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
    // The number of keys per warp.
    constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
    // The number of unrolled keys per warp.
    constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
    // The number of unrolled keys per ieration.
    // Never Used
    // constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

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

    // Decompose the thread index into warp and lane.
    // Never Used
    // auto const warp = tidx / WARP_SIZE;
    auto const lane = tidx % WARP_SIZE;

    int current_step_ctile_idx = kv_loop_length / timesteps_per_block;

    // CHECKLIST: Split-Point======================================================================================

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // [x] Restore the qk_values and qk_max_value for each head

    // [ ] If this code perform incorrectly, let each thread takes qk_max from qk_max_values.
    qk_max = lane == 0 ? params.qk_max_values[hi] : 0;

    // Broadcast to all the threads in the warp
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    // [x] Restore the qk_smem from params.qk_values
    auto const max_attention_window_size = params.max_attention_window_size;
    int const qk_values_offset = hi * max_attention_window_size;

#pragma unroll
    for (int out_i = tidx; out_i <= kv_loop_length; out_i += THREADS_PER_BLOCK)
    {
        qk_smem[out_i] = params.qk_values[qk_values_offset + out_i];
    }

    __syncthreads();

    // TODO_: Need to restore 'qk_current_smem' if we consider 'MULTI_BLOCK_FLAG'

    // [x] Restore the qk_values and qk_max_value for each head
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute the logits and start the sum.
    float sum = 0.f;

    // Each thread will handle one float (either qk_smem/logit).
    int const logit_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= logit_loop_end; ti += THREADS_PER_BLOCK)
    {

        int const time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

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

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

// Normalize the logits.
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float logit_scale = (FP8_KV_CACHE ? kv_scale_quant_orig_f : 1.0f);
#else
    float logit_scale = 1.f;
#endif // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float inv_sum = __fdividef(logit_scale, sum + 1.e-6f);

    int const normlization_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= normlization_loop_end; ti += THREADS_PER_BLOCK)
    {
        int const time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

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

    // CHECKLIST: Split-Point======================================================================================

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    auto const v_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
    // The value computed by this thread.
    auto const vo = v_idx.x;
    // The hidden dimensions computed by this particular thread.
    auto const vi = v_idx.y;

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
            auto const v_bias_offset = tensorrt_llm::common::flat_index2(hi_kv, vi, Dh);
            v_bias = *reinterpret_cast<V_vec_k const*>(&params.v_bias[v_bias_offset]);
        }

        if (DO_CROSS_ATTENTION)
        {
            *reinterpret_cast<V_vec_k*>(&bias_smem[vi]) = v_bias;
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Value cache loops.

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
        int const context_length
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
                if (POS_SHIFT && time_idx >= sink_token_len)
                {
                    // If one more block mode is enabled, we use the index in sequence as tokenIdx.
                    // Otherwise, we need to add the bubble length to the index
                    time_idx += shift_for_cyclic_kv;
                    if (enable_use_seq_idx_kv)
                    {
                        // Convert the token index in sequence to token index in V cache.
                        time_idx = kvCacheBuffer.getKVTokenIdx(time_idx);
                    }
                }
                int rowIdx = batch_idx * beam_width;

                int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                // The base pointer for the value in the cache buffer.
                Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                v_vec_cache[v_loop] = *reinterpret_cast<V_vec_m const*>(&v_cache_batch[inBlockIdx]);
            }

#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                V_vec_m v_vec = reinterpret_cast<V_vec_m*>(&v_vec_cache[v_loop])[0];

                int local_time_idx = ti + v_loop * V_PER_ITER;
                int time_idx = local_time_idx + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);

                bool const is_mask
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
            auto const generation_start_ti
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

                    if (POS_SHIFT && time_idx >= sink_token_len)
                    {
                        // If one more block mode is enabled, we use the index in sequence as tokenIdx.
                        // Otherwise, we need to add the bubble length to the index
                        time_idx += shift_for_cyclic_kv;
                        if (enable_use_seq_idx_kv)
                        {
                            // Convert the token index in sequence to token index in V cache.
                            time_idx = kvCacheBuffer.getKVTokenIdx(time_idx);
                        }
                    }

                    int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                    // The base pointer for the value in the cache buffer.
                    Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));
                    V_vec_m v_vec = reinterpret_cast<V_vec_m const*>(&v_cache_batch[inBlockIdx])[0];

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

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // One group of threads computes the product(s) for the current timestep.
    if (vo == kv_loop_length % V_PER_ITER && is_valid_vi && (!MULTI_BLOCK_FLAG || (c_tile == current_step_ctile_idx)))
    {
        int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(cyclic_tlength, hi_kv, Dh, vi);
        // The base pointer for the value in the cache buffer.
        Tcache* v_cache_base = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(batch_beam_idx, cyclic_tlength));

        V_vec_k v;
        if (DO_CROSS_ATTENTION)
        {
            v = vec_conversion<V_vec_k, V_vec_k>(*reinterpret_cast<V_vec_k const*>(&v_cache_base[inBlockIdx]));
        }
        else
        {
            // Trigger the loads from the V buffer.
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            auto const v_offset = tensorrt_llm::common::flat_index_strided3(batch_beam_idx, hi_kv, vi, v_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_k>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec_k>::value>::type;
                auto const v_scaling = params.qkv_scale_quant_orig[2];
                auto const v_quant
                    = *reinterpret_cast<Packed_Int8_t const*>(&reinterpret_cast<int8_t const*>(params.v)[v_offset]);

                convert_from_float(&v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else
            {
                v = *reinterpret_cast<V_vec_k const*>(&params.v[v_offset]);
            }
        }

        if (HANDLE_KV)
        {
            // Compute the V values with bias.
            v = add(v, v_bias);

            if (do_ia3)
            {
                v = mul<V_vec_k, V_vec_k, V_vec_k>(v,
                    *reinterpret_cast<V_vec_k const*>(
                        &params.ia3_value_weights[tensorrt_llm::common::flat_index2(ia3_ti_hi, vi, Dh)]));
            }
        }

        // Store the values with bias back to global memory in the cache for V.
        //*reinterpret_cast<V_vec_k*>(&v_cache[params.timestep*Dh]) = v;
        // For MQA/GQA mode, write only with the first Q head of each group per KV head.
        if (hi == (hi_kv * qhead_per_kv))
        {
            if (ENABLE_8BITS_KV_CACHE)
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

    // CHECKLIST: out => partial_out

    // Make sure we can start writing to shared memory.
    __syncthreads();

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
            out = add(*reinterpret_cast<V_vec_k const*>(&out_smem[vo * Dh + vi]), out);
        }
        __syncthreads();
    }

    // Quantized output only supports fp8 currently, which should be used together with FP8 Context FMHA.
    using Quantized_t = __nv_fp8_e4m3;
    using Quantized_vec = typename packed_type<__nv_fp8_e4m3, num_elems<V_vec_accum>::value>::type;
    auto const bhi = tensorrt_llm::common::flat_index2(batch_beam_idx, hi, num_heads);
    auto const bhi_seq_len_tile = bhi * params.seq_len_tile;
    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh))
    {
        auto const bhvi = tensorrt_llm::common::flat_index2(bhi, vi, Dh);
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
        if (!MULTI_BLOCK_FLAG)
        {
            if (write_attention_quant)
            {
                out = mul<V_vec_accum, float>(*params.attention_out_scale_orig_quant, out);
                Quantized_vec final_out;
                convert_to_fp8(&final_out, out);
                *reinterpret_cast<Quantized_vec*>(reinterpret_cast<Quantized_t*>(params.out) + bhvi) = final_out;
            }
            else
            {
                // This makes sure we have coalesced memory access.
                V_vec_k final_out;
                convert_from_float(&final_out, out);
                *reinterpret_cast<V_vec_k*>(static_cast<T*>(params.out) + bhvi) = final_out;
            }
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
#else  // MMHA_USE_FP32_ACCUM_FOR_OUT
        *reinterpret_cast<V_vec_accum*>(static_cast<T*>(params.out) + bhvi) = out;
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
                auto const thread_partial_sum = params.partial_sum[bhi_seq_len_tile + tidx];
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

            auto const o_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);

            // Init partial out for accumulation.
            V_vec_k zero_k;
            zero(zero_k);
            V_vec_k thread_accumulated_out = zero_k;

            // The hidden dimensions computed by this particular thread. (refer to vi)
            auto const oi = o_idx.y;

            // The partial output region this thread takes care of
            auto const oo = o_idx.x;

            // Each thread may handle more than one partial output.
            for (int tile_idx = o_idx.x; tile_idx < gridDim.z; tile_idx += V_PER_ITER)
            {
                // Load partial output
                int thread_partial_out_offset = tile_idx * params.batch_size * num_heads * params.hidden_size_per_head;
                // Load partial max (different to thread_partial_max since the threadIdx rule changes here)
                float thread_partial_max_for_out = params.partial_max[bhi_seq_len_tile + tile_idx];
                // Load the partial outputs.
                V_vec_k thread_partial_out
                    = *reinterpret_cast<V_vec_k const*>(&params.partial_out[thread_partial_out_offset + bhi * Dh + oi]);
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
                        = add(thread_accumulated_out, *reinterpret_cast<V_vec_k const*>(&out_oi_smem[oo * Dh + oi]));
                }
                __syncthreads();
            }

            ////////////////////
            // Final output O * inv_sum
            ////////////////////

            if (oo == 0 && (Dh == Dh_MAX || oi < Dh))
            {
                auto const inv_sum = __fdividef(
                    write_attention_quant ? *params.attention_out_scale_orig_quant : 1.f, final_sum + 1.e-6f);

                Tk inv_sum_compute;
                convert_from_float(&inv_sum_compute, inv_sum);

                thread_accumulated_out = mul<V_vec_k, Tk, V_vec_k>(inv_sum_compute, thread_accumulated_out);

                if (write_attention_quant)
                {
                    Quantized_vec final_out;
                    convert_to_fp8(&final_out, thread_accumulated_out);
                    *reinterpret_cast<Quantized_vec*>(reinterpret_cast<Quantized_t*>(params.out) + bhi * Dh + oi)
                        = final_out;
                }
                else
                {
                    *reinterpret_cast<V_vec_k*>(static_cast<T*>(params.out) + (bhi * Dh + oi)) = thread_accumulated_out;
                }
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

} // namespace mmha

} // namespace kernels
} // namespace tensorrt_llm
