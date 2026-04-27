// Prevent Windows min/max macros from conflicting with std::min/std::max
#define NOMINMAX

#include "common.cuh"
#include "dsv4.cuh"

#ifndef M_PI_F
#define M_PI_F 3.141592653589793238462643383279502884f
#endif

namespace {

constexpr int DSV4_HC_MAX = 16;

static __device__ __forceinline__ float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * exp2f(float(exp - 7));
}

static __device__ __forceinline__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 448.0f);

    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float val = dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

static __device__ __forceinline__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ __forceinline__ float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * logf(base));
}

static __device__ __forceinline__ void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    dims[0] = max(0.0f,         floorf(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceilf(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

static __device__ __forceinline__ void rope_yarn(float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale, float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

struct ggml_cuda_kargs_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ggml_cuda_kargs_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
};

struct ggml_cuda_kargs_dsv4_fp8_kv_quantize {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_rot;
};

struct ggml_cuda_kargs_dsv4_rope_tail {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_dims;
    int32_t  mode;
    int32_t  n_ctx_orig;
    int32_t  inverse;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    bool     src2;
};

static __global__ void kernel_dsv4_hc_split_sinkhorn(
        const ggml_cuda_kargs_dsv4_hc_split_sinkhorn args,
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int64_t) tid >= args.n_rows) {
        return;
    }

    const int HC = args.n_hc;
    if (HC <= 0 || HC > DSV4_HC_MAX) {
        return;
    }

    const float * mix = mixes + ((int64_t) tid) * args.mix_hc;
    float * out = dst + ((int64_t) tid) * args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[DSV4_HC_MAX * DSV4_HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const int off = 2 * HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc * HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc * HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC * HC; ++i) {
        out[2 * HC + i] = c[i];
    }
}

static __global__ void kernel_dsv4_hc_expand(
        const ggml_cuda_kargs_dsv4_hc_expand args,
        const char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        char * dst) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    const float block_v = *((const float *) (block_out + d * args.nb_block0 + t * args.nb_block1));
    const float post_v  = *((const float *) (post      + dst_hc * args.nb_post0 + t * args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((const float *) (comb     + dst_hc * args.nb_comb0 + src_hc * args.nb_comb1 + t * args.nb_comb2));
        const float res_v  = *((const float *) (residual + d       * args.nb_res0  + src_hc * args.nb_res1  + t * args.nb_res2));
        acc += comb_v * res_v;
    }

    *((float *) (dst + d * args.nb0 + dst_hc * args.nb1 + t * args.nb2)) = acc;
}

static __global__ void kernel_dsv4_fp8_kv_quantize(
        const ggml_cuda_kargs_dsv4_fp8_kv_quantize args,
        const char * src0,
        char * dst) {
    __shared__ float scratch[64];

    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    const int row = blockIdx.x;
    if ((int64_t) row >= n_rows) {
        return;
    }

    const int tid = threadIdx.x;

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    const char * src_base = src0 + i1 * args.nb01 + i2 * args.nb02 + i3 * args.nb03;
    char * dst_base = dst  + i1 * args.nb1  + i2 * args.nb2  + i3 * args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64) {
            v = *((const float *) (src_base + (off + tid) * args.nb00));
            scratch[tid] = fabsf(v);
        }
        __syncthreads();

        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (tid < 64) {
            const float q = dsv4_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            *((float *) (dst_base + (off + tid) * args.nb0)) = q;
        }
        __syncthreads();
    }

    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((float *) (dst_base + i * args.nb0)) = *((const float *) (src_base + i * args.nb00));
    }
}

static __global__ void kernel_dsv4_rope_tail_f32(
        const ggml_cuda_kargs_dsv4_rope_tail args,
        const char * src0,
        const char * src1,
        const char * src2,
        char * dst) {
    const int i1 = blockIdx.z;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.x;

    const int tid = threadIdx.x;

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    const int32_t * pos = (const int32_t *) src1;

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.0f / args.n_dims;
    const bool is_neox = args.mode == 2;

    for (int i0 = tid; i0 < args.ne00; i0 += blockDim.x) {
        const char * src_base = src0 + i3 * args.nb03 + i2 * args.nb02 + i1 * args.nb01;
        char * dst_base = dst  + i3 * args.nb3  + i2 * args.nb2  + i1 * args.nb1;

        if (i0 < n_nope) {
            *((float *) (dst_base + i0 * args.nb0)) = *((const float *) (src_base + i0 * args.nb00));
            continue;
        }

        const int r = i0 - n_nope;
        if (is_neox) {
            const int n_half = args.n_dims / 2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2 * ic;
            const float theta = theta_base * powf(args.freq_base, inv_ndims * rel_i0);
            const float freq_factor = args.src2 ? ((const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta / freq_factor, args.freq_scale, corr_dims, rel_i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *((const float *) (src_base + j0 * args.nb00));
            const float x1 = *((const float *) (src_base + j1 * args.nb00));

            *((float *) (dst_base + j0 * args.nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * args.nb0)) = x0 * sin_theta + x1 * cos_theta;
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r / 2;
            const float theta = theta_base * powf(args.freq_base, inv_ndims * r);
            const float freq_factor = args.src2 ? ((const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta / freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *((const float *) (src_base + j0 * args.nb00));
            const float x1 = *((const float *) (src_base + j1 * args.nb00));

            *((float *) (dst_base + j0 * args.nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * args.nb0)) = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

} // namespace

bool ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1);
    GGML_ASSERT(src0->ne[3] == 1);

    const int32_t n_hc           = ggml_get_op_params_i32(dst, 0);
    const int32_t sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps              = ggml_get_op_params_f32(dst, 2);

    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne00 = src0->ne[0];

    const int64_t n_rows = ne01 * ne02 * ne03;

    const float * mixes_d = (const float *) src0->data;
    const float * scale_d = (const float *) src1->data;
    const float * base_d  = (const float *) src2->data;
    float * dst_d = (float *) dst->data;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_rows));
    const int n_tg = (n_rows + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_split_sinkhorn args = {
        /*.n_hc            =*/ n_hc,
        /*.sinkhorn_iters  =*/ sinkhorn_iters,
        /*.n_rows          =*/ n_rows,
        /*.mix_hc          =*/ ne00,
        /*.nb01            =*/ src0->nb[1],
        /*.nb1             =*/ dst->nb[1],
        /*.eps             =*/ eps,
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_hc_split_sinkhorn<<<n_tg, nth, 0, stream>>>(args, mixes_d, scale_d, base_d, dst_d);

    return true;
}

bool ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * block_out = dst->src[0];
    ggml_tensor * residual  = dst->src[1];
    ggml_tensor * post      = dst->src[2];
    ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];

    const int64_t n_elem = ne0 * ne1 * ne2;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_elem));
    const int n_tg = (n_elem + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_expand args = {
        /*.n_embd    =*/ ne0,
        /*.n_hc      =*/ ne1,
        /*.n_tokens  =*/ ne2,
        /*.nb_block0 =*/ block_out->nb[0],
        /*.nb_block1 =*/ block_out->nb[1],
        /*.nb_res0   =*/ residual->nb[0],
        /*.nb_res1   =*/ residual->nb[1],
        /*.nb_res2   =*/ residual->nb[2],
        /*.nb_post0  =*/ post->nb[0],
        /*.nb_post1  =*/ post->nb[1],
        /*.nb_comb0  =*/ comb->nb[0],
        /*.nb_comb1  =*/ comb->nb[1],
        /*.nb_comb2  =*/ comb->nb[2],
        /*.nb0       =*/ dst->nb[0],
        /*.nb1       =*/ dst->nb[1],
        /*.nb2       =*/ dst->nb[2],
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_hc_expand<<<n_tg, nth, 0, stream>>>(
        args,
        (const char *) block_out->data,
        (const char *) residual->data,
        (const char *) post->data,
        (const char *) comb->data,
        (char *) dst->data);

    return true;
}

bool ggml_cuda_op_dsv4_rope_tail_supported(void) {
    // Supported: F32 input/output
    return true;
}

bool ggml_cuda_op_dsv4_hc_split_sinkhorn_supported(void) {
    // Placeholder - kernel not yet implemented
    return false;
}

bool ggml_cuda_op_dsv4_hc_expand_supported(void) {
    // Placeholder - kernel not yet implemented
    return false;
}

bool ggml_cuda_op_dsv4_fp8_kv_quantize_supported(void) {
    // Supported: F32 input/output
    return true;
}

bool ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t n_rot = ggml_get_op_params_i32(dst, 0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t n_rows = ne01 * ne02 * ne03;

    ggml_cuda_kargs_dsv4_fp8_kv_quantize args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ src0->nb[0],
        /*.nb01 =*/ src0->nb[1],
        /*.nb02 =*/ src0->nb[2],
        /*.nb03 =*/ src0->nb[3],
        /*.nb0  =*/ dst->nb[0],
        /*.nb1  =*/ dst->nb[1],
        /*.nb2  =*/ dst->nb[2],
        /*.nb3  =*/ dst->nb[3],
        /*.n_rot =*/ n_rot,
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_fp8_kv_quantize<<<n_rows, 64, 0, stream>>>(
        args,
        (const char *) src0->data,
        (char *) dst->data);

    return true;
}

bool ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t n_dims     = ggml_get_op_params_i32(dst, 0);
    const int32_t mode       = ggml_get_op_params_i32(dst, 1);
    const int32_t n_ctx_orig = ggml_get_op_params_i32(dst, 2);
    const int32_t inverse    = ggml_get_op_params_i32(dst, 3);

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) dst->op_params + 9, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, ne00));

    ggml_cuda_kargs_dsv4_rope_tail args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ src0->nb[0],
        /*.nb01        =*/ src0->nb[1],
        /*.nb02        =*/ src0->nb[2],
        /*.nb03        =*/ src0->nb[3],
        /*.nb0         =*/ dst->nb[0],
        /*.nb1         =*/ dst->nb[1],
        /*.nb2         =*/ dst->nb[2],
        /*.nb3         =*/ dst->nb[3],
        /*.n_dims      =*/ n_dims,
        /*.mode        =*/ mode,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.inverse     =*/ inverse,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /*.src2        =*/ src2 != nullptr,
    };

    const cudaStream_t stream = ctx.stream();

    dim3 grid(ne03, ne02, ne01);

    kernel_dsv4_rope_tail_f32<<<grid, nth, 0, stream>>>(
        args,
        (const char *) src0->data,
        (const char *) src1->data,
        src2 ? (const char *) src2->data : (const char *) src0->data,
        (char *) dst->data);

    return true;
}