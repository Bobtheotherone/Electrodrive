// electrodrive/core/bem_near_cuda.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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

// Expected signature implemented in your .cu file.
//
// torch::Tensor bem_near_quadrature_matvec_cuda(
//     const torch::Tensor& V_far,
//     const torch::Tensor& sigma,
//     const torch::Tensor& centroids,
//     const torch::Tensor& areas,
//     const torch::Tensor& panel_vertices,
//     const torch::Tensor& near_pairs,
//     double K_E,
//     int64_t quad_order);
torch::Tensor bem_near_quadrature_matvec_cuda(
    const torch::Tensor& V_far,
    const torch::Tensor& sigma,
    const torch::Tensor& centroids,
    const torch::Tensor& areas,
    const torch::Tensor& panel_vertices,
    const torch::Tensor& near_pairs,
    double K_E,
    int64_t quad_order);

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

    auto V_out = bem_near_quadrature_matvec_cuda(
        V_far_c,
        sigma_c,
        centroids_c,
        areas_c,
        panel_vertices_c,
        near_pairs_c,
        K_E,
        quad_order);

    TORCH_CHECK(
        V_out.device() == dev,
        "bem_near_quadrature_matvec_cuda returned tensor on wrong device");
    TORCH_CHECK(
        V_out.sizes() == V_far_c.sizes(),
        "bem_near_quadrature_matvec_cuda must return tensor with shape matching V_far");

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
quad_order:      Quadrature order (e.g. 2).

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
