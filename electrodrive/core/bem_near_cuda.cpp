// electrodrive/core/bem_near_cuda.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

namespace {

// Basic sanity checks for CUDA float tensors
void ensure_cuda_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        t.dtype() == torch::kFloat32 || t.dtype() == torch::kFloat64,
        name, " must have dtype float32 or float64");
}

void ensure_cuda_int64(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kInt64, name, " must have dtype int64");
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// CUDA implementation (defined in bem_near_cuda_kernel.cu)
// -----------------------------------------------------------------------------
//
// The .cu file implements a low-level routine that expects geometry, near-pair
// indices, and a pre-built reference quadrature rule on the unit triangle, and
// returns a *correction* vector V_corr [N]:
//
//   Tensor bem_near_quadrature_matvec_cuda(
//       Tensor centroids,   // [N,3], CUDA
//       Tensor areas,       // [N]
//       Tensor sigma,       // [N]
//       Tensor vertices,    // [N,3,3]
//       Tensor near_pairs,  // [P,2], int32 or int64
//       Tensor ref_pts,     // [Q,2] on unit triangle
//       Tensor ref_w,       // [Q] with sum_k ref_w[k] = 1
//       double K_E_double);
//
// The wrapper below builds (ref_pts, ref_w) based on quad_order, calls this
// low-level CUDA function to get V_corr, and then returns V_out = V_far + V_corr.

at::Tensor bem_near_quadrature_matvec_cuda(
    at::Tensor centroids,   // [N,3], float32 or float64, CUDA
    at::Tensor areas,       // [N]
    at::Tensor sigma,       // [N]
    at::Tensor vertices,    // [N,3,3]
    at::Tensor near_pairs,  // [P,2], int32 or int64
    at::Tensor ref_pts,     // [Q,2]
    at::Tensor ref_w,       // [Q]
    double K_E_double);

// -----------------------------------------------------------------------------
// Thin checked wrapper exposed to Python
// -----------------------------------------------------------------------------

torch::Tensor bem_near_quadrature_matvec(
    torch::Tensor V_far,
    torch::Tensor sigma,
    torch::Tensor centroids,
    torch::Tensor areas,
    torch::Tensor panel_vertices,
    torch::Tensor near_pairs,
    double K_E,
    int64_t quad_order) {

    TORCH_CHECK(
        V_far.device().is_cuda(),
        "bem_near_quadrature_matvec: V_far must be a CUDA tensor");

    // Device consistency checks
    auto dev = V_far.device();
    TORCH_CHECK(sigma.device() == dev, "sigma must be on same device as V_far");
    TORCH_CHECK(centroids.device() == dev, "centroids must be on same device as V_far");
    TORCH_CHECK(areas.device() == dev, "areas must be on same device as V_far");
    TORCH_CHECK(panel_vertices.device() == dev, "panel_vertices must be on same device as V_far");
    TORCH_CHECK(near_pairs.device().is_cuda(), "near_pairs must be a CUDA tensor");

    // Shape sanity
    TORCH_CHECK(
        V_far.dim() == 1,
        "V_far must be a 1D vector of shape [N], got dim=", V_far.dim());
    TORCH_CHECK(
        centroids.dim() == 2 && centroids.size(1) == 3,
        "centroids must have shape [N, 3]");
    TORCH_CHECK(
        areas.dim() == 1,
        "areas must have shape [N]");
    TORCH_CHECK(
        panel_vertices.dim() == 3 &&
        panel_vertices.size(1) == 3 &&
        panel_vertices.size(2) == 3,
        "panel_vertices must have shape [N, 3, 3]");
    TORCH_CHECK(
        near_pairs.dim() == 2 && near_pairs.size(1) == 2,
        "near_pairs must have shape [P, 2]");

    auto N = centroids.size(0);
    TORCH_CHECK(
        areas.size(0) == N &&
        panel_vertices.size(0) == N &&
        sigma.size(0) == N &&
        V_far.size(0) == N,
        "Inconsistent panel dimension N between inputs: "
        "centroids.size(0)=", N,
        ", areas.size(0)=", areas.size(0),
        ", panel_vertices.size(0)=", panel_vertices.size(0),
        ", sigma.size(0)=", sigma.size(0),
        ", V_far.size(0)=", V_far.size(0));

    // Dtype / layout checks
    ensure_cuda_float(V_far, "V_far");
    ensure_cuda_float(sigma, "sigma");
    ensure_cuda_float(centroids, "centroids");
    ensure_cuda_float(areas, "areas");
    ensure_cuda_float(panel_vertices, "panel_vertices");
    ensure_cuda_int64(near_pairs, "near_pairs");

    TORCH_CHECK(
        V_far.dtype() == sigma.dtype(),
        "V_far and sigma must have the same dtype");
    TORCH_CHECK(
        V_far.dtype() == centroids.dtype(),
        "V_far and centroids must have the same dtype");
    TORCH_CHECK(
        V_far.dtype() == areas.dtype(),
        "V_far and areas must have the same dtype");
    TORCH_CHECK(
        V_far.dtype() == panel_vertices.dtype(),
        "V_far and panel_vertices must have the same dtype");

    // Guard the device / stream
    const c10::cuda::CUDAGuard device_guard(dev);

    // Make sure everything is contiguous before passing to the CUDA core
    auto V_far_c          = V_far.contiguous();
    auto sigma_c          = sigma.contiguous();
    auto centroids_c      = centroids.contiguous();
    auto areas_c          = areas.contiguous();
    auto panel_vertices_c = panel_vertices.contiguous();
    auto near_pairs_c     = near_pairs.contiguous();

    // -------------------------------------------------------------------------
    // Build reference quadrature rule (ref_pts, ref_w) on the unit triangle.
    // We support quad_order 1 and 2 using standard symmetric rules.
    //
    //  - Order 1: 1-point rule at barycenter (1/3,1/3), weight 1.
    //  - Order 2: 3-point rule:
    //        (1/6, 1/6), (2/3, 1/6), (1/6, 2/3), each weight 1/3.
    // -------------------------------------------------------------------------
    TORCH_CHECK(
        quad_order == 1 || quad_order == 2,
        "bem_near_quadrature_matvec: only quad_order 1 or 2 are supported, got ",
        quad_order);

    const auto opts = centroids_c.options();
    const auto opts_cpu = opts.device(torch::kCPU);
    int64_t Q = (quad_order == 1) ? 1 : 3;

    // Build quadrature rule on CPU to avoid host-side writes into CUDA memory,
    // then transfer the tiny tensors to the target device.
    auto ref_pts_cpu = torch::empty({Q, 2}, opts_cpu);
    auto ref_w_cpu   = torch::empty({Q},     opts_cpu);

    AT_DISPATCH_FLOATING_TYPES(
        centroids_c.scalar_type(),
        "build_ref_triangle_quadrature",
        [&] {
            using scalar_t = scalar_t;
            auto ref_pts_ptr = ref_pts_cpu.data_ptr<scalar_t>();
            auto ref_w_ptr   = ref_w_cpu.data_ptr<scalar_t>();

            if (quad_order == 1) {
                // Single-point rule: barycenter with weight 1.
                ref_pts_ptr[0 * 2 + 0] = static_cast<scalar_t>(1.0 / 3.0);
                ref_pts_ptr[0 * 2 + 1] = static_cast<scalar_t>(1.0 / 3.0);
                ref_w_ptr[0]           = static_cast<scalar_t>(1.0);
            } else {
                // 3-point symmetric rule on unit triangle.
                // Points:
                //  p0 = (1/6, 1/6)
                //  p1 = (2/3, 1/6)
                //  p2 = (1/6, 2/3)
                // Weights: each 1/3, sum to 1.
                ref_pts_ptr[0 * 2 + 0] = static_cast<scalar_t>(1.0 / 6.0);
                ref_pts_ptr[0 * 2 + 1] = static_cast<scalar_t>(1.0 / 6.0);

                ref_pts_ptr[1 * 2 + 0] = static_cast<scalar_t>(2.0 / 3.0);
                ref_pts_ptr[1 * 2 + 1] = static_cast<scalar_t>(1.0 / 6.0);

                ref_pts_ptr[2 * 2 + 0] = static_cast<scalar_t>(1.0 / 6.0);
                ref_pts_ptr[2 * 2 + 1] = static_cast<scalar_t>(2.0 / 3.0);

                const scalar_t w = static_cast<scalar_t>(1.0 / 3.0);
                ref_w_ptr[0] = w;
                ref_w_ptr[1] = w;
                ref_w_ptr[2] = w;
            }
        });

    auto ref_pts = ref_pts_cpu.to(opts.device());
    auto ref_w   = ref_w_cpu.to(opts.device());

    // If there are no near pairs, nothing to correct.
    if (near_pairs_c.size(0) == 0) {
        return V_far_c;
    }

    // Call the CUDA implementation to get the near-field correction V_corr [N].
    auto V_corr = bem_near_quadrature_matvec_cuda(
        centroids_c,
        areas_c,
        sigma_c,
        panel_vertices_c,
        near_pairs_c,
        ref_pts,
        ref_w,
        K_E);

    TORCH_CHECK(
        V_corr.device() == dev,
        "bem_near_quadrature_matvec_cuda returned tensor on wrong device");
    TORCH_CHECK(
        V_corr.sizes() == V_far_c.sizes(),
        "bem_near_quadrature_matvec_cuda must return tensor with shape [N]");

    auto V_out = V_far_c + V_corr;
    return V_out;
}

// -----------------------------------------------------------------------------
// pybind11 module
// -----------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated near-field quadrature correction for BEM matvec";

    m.def(
        "near_quadrature_matvec",
        &bem_near_quadrature_matvec,
        R"doc(
Apply near-field quadrature correction to a BEM matvec on CUDA.

Args
----
V_far:           [N] CUDA tensor (initial far-field result).
sigma:           [N] CUDA tensor of panel charges.
centroids:       [N, 3] CUDA tensor of panel centroids.
areas:           [N] CUDA tensor of panel areas.
panel_vertices:  [N, 3, 3] CUDA tensor of triangle vertex positions.
near_pairs:      [P, 2] int64 CUDA tensor of (i, j) near interaction indices.
K_E:             Coulomb constant (double).
quad_order:      Quadrature order (e.g. 1 or 2).

Returns
-------
Corrected potential vector with the same shape as V_far.
)doc",
        pybind11::arg("V_far"),
        pybind11::arg("sigma"),
        pybind11::arg("centroids"),
        pybind11::arg("areas"),
        pybind11::arg("panel_vertices"),
        pybind11::arg("near_pairs"),
        pybind11::arg("K_E"),
        pybind11::arg("quad_order"));
}
