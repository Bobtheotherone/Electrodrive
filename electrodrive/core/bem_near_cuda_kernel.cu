// bem_near_cuda_kernel.cu
//
// Near-field quadrature corrections for single-layer BEM matvec on CUDA GPUs.
// One thread per (target panel i, source panel j) near pair.
//
// For each near pair (i, j):
//   1) Build physical quadrature nodes on source triangle j from reference
//      (u, v) nodes on the unit triangle using an affine map.
//   2) Evaluate the Laplace single-layer kernel at target centroid C_i.
//   3) Compute I_near = Sum_k [ K_E / |C_i - y_k| * (w_ref[k] * A_j) ].
//   4) Compute the far-field lumped approximation
//          I_far = K_E * A_j / |C_i - C_j|.
//   5) Add the correction sigma_j * (I_near - I_far) to the potential at i.
//
// The host code is expected to:
//   - Build near_pairs [P, 2] on CPU (int32), where each row is (i, j).
//   - Build a reference triangle quadrature rule (ref_pts [Q,2], ref_w [Q])
//     on the *unit* triangle with Sum_k ref_w[k] = 1.
//   - Allocate a zero-initialized V_corr [N] on CUDA and pass it in.
//
// This file is written as a PyTorch CUDA extension, but the kernels
// themselves are plain CUDA C++ and can be reused in other contexts.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

using at::Tensor;

namespace {

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

template <typename scalar_t>
struct Vec3 {
    scalar_t x, y, z;

    __device__ __forceinline__ Vec3() = default;

    __device__ __forceinline__ Vec3(scalar_t x_, scalar_t y_, scalar_t z_)
        : x(x_), y(y_), z(z_) {}

    __device__ __forceinline__ Vec3 operator-(const Vec3& other) const {
        return Vec3{x - other.x, y - other.y, z - other.z};
    }

    __device__ __forceinline__ Vec3 operator+(const Vec3& other) const {
        return Vec3{x + other.x, y + other.y, z + other.z};
    }

    __device__ __forceinline__ Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t norm(const Vec3<scalar_t>& v) {
    // Use sqrt here (not rsqrt) for accuracy; this kernel is on the critical
    // accuracy path for the BEM operator.
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Load centroid i from [N,3] row-major array.
template <typename scalar_t>
__device__ __forceinline__ Vec3<scalar_t>
load_centroid(const scalar_t* __restrict__ centroids, int i) {
    const scalar_t* base = centroids + 3 * static_cast<long long>(i);
    return Vec3<scalar_t>{base[0], base[1], base[2]};
}

// Load triangle j from [N,3,3] row-major as [N][3][3] -> 9 contiguous scalars.
template <typename scalar_t>
__device__ __forceinline__ void
load_triangle(const scalar_t* __restrict__ vertices,
              int j,
              Vec3<scalar_t>& v0,
              Vec3<scalar_t>& v1,
              Vec3<scalar_t>& v2) {
    const scalar_t* base = vertices + 9 * static_cast<long long>(j);
    v0 = Vec3<scalar_t>{base[0], base[1], base[2]};
    v1 = Vec3<scalar_t>{base[3], base[4], base[5]};
    v2 = Vec3<scalar_t>{base[6], base[7], base[8]};
}

// Simple clamp helper for positive values.
template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_min_pos(scalar_t x, scalar_t lo) {
    return x < lo ? lo : x;
}

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------

// We tune launch bounds for Blackwell-class GPUs. 256 threads/block with at
// least two resident blocks/SM gives good occupancy while leaving registers
// for heavy math in the quadrature loop.
constexpr int BEM_NEAR_BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ __launch_bounds__(BEM_NEAR_BLOCK_SIZE, 2)
void bem_near_quad_matvec_kernel(
    const long long num_pairs,
    const int2* __restrict__ near_pairs,   // [P]
    const scalar_t* __restrict__ centroids, // [N,3]
    const scalar_t* __restrict__ areas,     // [N]
    const scalar_t* __restrict__ sigma,     // [N]
    const scalar_t* __restrict__ vertices,  // [N,3,3] -> [N,9]
    const scalar_t* __restrict__ ref_pts,   // [Q,2] (u,v) on unit triangle
    const scalar_t* __restrict__ ref_w,     // [Q], sum_k ref_w[k] = 1
    const int Q,
    const scalar_t K_E,
    scalar_t* __restrict__ V_corr           // [N], zero-initialized
) {
    extern __shared__ unsigned char smem_raw[];
    scalar_t* sh_u = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* sh_v = sh_u + Q;
    scalar_t* sh_w = sh_v + Q;

    // Cooperative load of reference quadrature nodes and weights into shared
    // memory. Q is small (<= ~7 for current config), so this is cheap and
    // keeps everything in fast memory for the inner loop.
    for (int idx = threadIdx.x; idx < Q; idx += blockDim.x) {
        sh_u[idx] = ref_pts[2 * idx + 0];
        sh_v[idx] = ref_pts[2 * idx + 1];
        sh_w[idx] = ref_w[idx];
    }
    __syncthreads();

    const long long global_tid =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_tid >= num_pairs) {
        return;
    }

    // Decode near pair (i, j).
    const int2 pair = near_pairs[global_tid];
    const int i = pair.x;
    const int j = pair.y;

    // Load primitive data.
    const scalar_t Aj = areas[j];
    const scalar_t sig_j = sigma[j];

    if (Aj <= scalar_t(0) || sig_j == scalar_t(0)) {
        return;
    }

    const Vec3<scalar_t> Ci = load_centroid(centroids, i);
    const Vec3<scalar_t> Cj = load_centroid(centroids, j);

    // Far-field lumped approximation for the same pair.
    const Vec3<scalar_t> d_far = Vec3<scalar_t>{
        Ci.x - Cj.x,
        Ci.y - Cj.y,
        Ci.z - Cj.z
    };

    const scalar_t eps_r = static_cast<scalar_t>(1e-12);
    scalar_t r_far = norm(d_far);
    r_far = clamp_min_pos(r_far, eps_r);
    const scalar_t I_far = K_E * Aj / r_far;

    // Triangle geometry for source panel j.
    Vec3<scalar_t> v0, v1, v2;
    load_triangle(vertices, j, v0, v1, v2);

    const Vec3<scalar_t> e1 = Vec3<scalar_t>{
        v1.x - v0.x,
        v1.y - v0.y,
        v1.z - v0.z
    };
    const Vec3<scalar_t> e2 = Vec3<scalar_t>{
        v2.x - v0.x,
        v2.y - v0.y,
        v2.z - v0.z
    };

    // Near-field quadrature.
    scalar_t I_near = scalar_t(0);

    // The loop bound Q is typically small (1 or 3 for the supported near
    // orders). We unroll up to a fixed MAX_Q and guard by (k < Q).
#pragma unroll
    for (int k = 0; k < 64; ++k) {
        if (k >= Q) break;

        const scalar_t u = sh_u[k];
        const scalar_t v = sh_v[k];
        const scalar_t wk = sh_w[k];

        // y = v0 + u * e1 + v * e2
        const Vec3<scalar_t> y = Vec3<scalar_t>{
            v0.x + u * e1.x + v * e2.x,
            v0.y + u * e1.y + v * e2.y,
            v0.z + u * e1.z + v * e2.z
        };

        const Vec3<scalar_t> d = Vec3<scalar_t>{
            Ci.x - y.x,
            Ci.y - y.y,
            Ci.z - y.z
        };

        scalar_t r = norm(d);
        r = clamp_min_pos(r, eps_r);

        // Physical weight: Aj scales the unit-triangle weights to the actual
        // physical area of the source panel.
        const scalar_t w_phys = wk * Aj;

        I_near += (K_E / r) * w_phys;
    }

    const scalar_t delta = sig_j * (I_near - I_far);

    // On Volta+ architectures (including Blackwell), nvcc performs
    // warp-aggregated atomics for many common patterns automatically, so
    // this simple atomicAdd still compiles to an efficient aggregated update
    // sequence in SASS on modern GPUs.
    atomicAdd(&V_corr[i], delta);
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Public launcher (PyTorch API)
// -----------------------------------------------------------------------------

Tensor bem_near_quadrature_matvec_cuda(
    Tensor centroids,   // [N,3], float32 or float64, CUDA
    Tensor areas,       // [N]
    Tensor sigma,       // [N]
    Tensor vertices,    // [N,3,3]
    Tensor near_pairs,  // [P,2], int32 or int64
    Tensor ref_pts,     // [Q,2], float32/float64 (same dtype as centroids)
    Tensor ref_w,       // [Q]
    double K_E_double
) {
    TORCH_CHECK(centroids.is_cuda(), "centroids must be a CUDA tensor");
    TORCH_CHECK(areas.is_cuda(),     "areas must be a CUDA tensor");
    TORCH_CHECK(sigma.is_cuda(),     "sigma must be a CUDA tensor");
    TORCH_CHECK(vertices.is_cuda(),  "vertices must be a CUDA tensor");
    TORCH_CHECK(near_pairs.is_cuda(),"near_pairs must be a CUDA tensor");
    TORCH_CHECK(ref_pts.is_cuda(),   "ref_pts must be a CUDA tensor");
    TORCH_CHECK(ref_w.is_cuda(),     "ref_w must be a CUDA tensor");

    TORCH_CHECK(centroids.dim() == 2 && centroids.size(1) == 3,
                "centroids must have shape [N,3]");
    TORCH_CHECK(vertices.dim() == 3 &&
                    vertices.size(1) == 3 && vertices.size(2) == 3,
                "vertices must have shape [N,3,3]");
    TORCH_CHECK(areas.dim() == 1 && areas.size(0) == centroids.size(0),
                "areas must have shape [N]");
    TORCH_CHECK(sigma.dim() == 1 && sigma.size(0) == centroids.size(0),
                "sigma must have shape [N]");
    TORCH_CHECK(near_pairs.dim() == 2 && near_pairs.size(1) == 2,
                "near_pairs must have shape [P,2]");
    TORCH_CHECK(ref_pts.dim() == 2 && ref_pts.size(1) == 2,
                "ref_pts must have shape [Q,2]");
    TORCH_CHECK(ref_w.dim() == 1 && ref_w.size(0) == ref_pts.size(0),
                "ref_w must have shape [Q] and match ref_pts.shape[0]");

    // Enforce a consistent dtype across scalar inputs.
    TORCH_CHECK(centroids.scalar_type() == vertices.scalar_type(),
                "centroids and vertices dtypes must match");
    TORCH_CHECK(centroids.scalar_type() == areas.scalar_type(),
                "centroids and areas dtypes must match");
    TORCH_CHECK(centroids.scalar_type() == sigma.scalar_type(),
                "centroids and sigma dtypes must match");
    TORCH_CHECK(centroids.scalar_type() == ref_pts.scalar_type(),
                "centroids and ref_pts dtypes must match");
    TORCH_CHECK(centroids.scalar_type() == ref_w.scalar_type(),
                "centroids and ref_w dtypes must match");

    auto opts = centroids.options();
    const auto N = centroids.size(0);
    const auto P = near_pairs.size(0);
    const auto Q = ref_w.size(0);

    // Hard safety cap: this kernel is tuned for small triangle rules.
    // The current BEM config only uses orders 1 and 2, which correspond
    // to very small Q (<= 7). If you increase the quadrature order in
    // Python, also bump MAX_Q and recompile.
    constexpr int MAX_Q = 64;
    TORCH_CHECK(Q <= MAX_Q,
                "bem_near_quadrature_matvec_cuda supports at most ", MAX_Q,
                " quadrature nodes, got Q=", Q);

    if (P == 0) {
        // Nothing to do; return a zero correction.
        return at::zeros({N}, opts);
    }

    // Make everything contiguous on device.
    centroids = centroids.contiguous();
    areas     = areas.contiguous();
    sigma     = sigma.contiguous();
    vertices  = vertices.contiguous();
    ref_pts   = ref_pts.contiguous();
    ref_w     = ref_w.contiguous();

    // near_pairs must be int32 [P,2] for the kernel (int2). If it is int64,
    // we cast once on the device.
    Tensor near_pairs_i32;
    if (near_pairs.scalar_type() == at::kInt) {
        near_pairs_i32 = near_pairs.contiguous();
    } else {
        near_pairs_i32 = near_pairs.to(at::kInt);
    }

    Tensor V_corr = at::zeros({N}, opts);

    const int threads = BEM_NEAR_BLOCK_SIZE;
    const int blocks = static_cast<int>((P + threads - 1) / threads);

    // Dispatch on dtype.
    AT_DISPATCH_FLOATING_TYPES(centroids.scalar_type(),
                               "bem_near_quadrature_matvec_cuda",
                               [&] {
        // Correct shared-memory size for the chosen scalar_t.
        const size_t shmem =
            static_cast<size_t>(3 * Q) * sizeof(scalar_t);

        auto stream = at::cuda::getCurrentCUDAStream();

        const int2* near_pairs_ptr =
            reinterpret_cast<const int2*>(near_pairs_i32.data_ptr<int>());

        bem_near_quad_matvec_kernel<scalar_t>
            <<<blocks, threads, shmem, stream>>>(
                static_cast<long long>(P),
                near_pairs_ptr,
                centroids.data_ptr<scalar_t>(),
                areas.data_ptr<scalar_t>(),
                sigma.data_ptr<scalar_t>(),
                vertices.data_ptr<scalar_t>(),
                ref_pts.data_ptr<scalar_t>(),
                ref_w.data_ptr<scalar_t>(),
                static_cast<int>(Q),
                static_cast<scalar_t>(K_E_double),
                V_corr.data_ptr<scalar_t>());
    });

    // Check for launch errors.
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "bem_near_quad_matvec_kernel launch failed: ",
                cudaGetErrorString(err));

    return V_corr;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "bem_near_quadrature_matvec",
        &bem_near_quadrature_matvec_cuda,
        "Near-field quadrature correction for single-layer BEM matvec (CUDA)");
}
