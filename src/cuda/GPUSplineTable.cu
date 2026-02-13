/**
 * @file GPUSplineTable.cu
 * @brief Implementation of GPU spline table management.
 */

#include "cuda/GPUSplineTable.h"
#include <cstring>
#include <algorithm>

#ifdef GOLLUMFIT_USE_CUDA

namespace gollumfit {
namespace gpu {

//==============================================================================
// GPUSplineManager Implementation
//==============================================================================

GPUSplineManager::GPUSplineManager() {}

GPUSplineManager::~GPUSplineManager() {
    clear();
}

GPUSplineManager::GPUSplineManager(GPUSplineManager&& other) noexcept
    : hostSplines_(std::move(other.hostSplines_)),
      deviceSplines_(std::move(other.deviceSplines_)) {
    other.hostSplines_.clear();
    other.deviceSplines_.clear();
}

GPUSplineManager& GPUSplineManager::operator=(GPUSplineManager&& other) noexcept {
    if (this != &other) {
        clear();
        hostSplines_ = std::move(other.hostSplines_);
        deviceSplines_ = std::move(other.deviceSplines_);
        other.hostSplines_.clear();
        other.deviceSplines_.clear();
    }
    return *this;
}

void GPUSplineManager::freeSpline(GPUSplineTable& spline) {
    // Free knot vectors
    for (int d = 0; d < spline.ndim; ++d) {
        if (spline.knots[d]) {
            cudaFree(spline.knots[d]);
            spline.knots[d] = nullptr;
        }
    }

    // Free coefficients
    if (spline.coeffs) {
        cudaFree(spline.coeffs);
        spline.coeffs = nullptr;
    }
}

void GPUSplineManager::clear() {
    // Free device memory for each spline
    for (auto& spline : hostSplines_) {
        freeSpline(spline);
    }

    // Free device structure pointers
    for (auto* ptr : deviceSplines_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    hostSplines_.clear();
    deviceSplines_.clear();
}

int GPUSplineManager::uploadSpline(
    int ndim,
    const int* orders,
    const std::vector<std::vector<double>>& knots,
    const std::vector<double>& coeffs,
    const int* nknots
) {
    if (ndim < 1 || ndim > 3) {
        throw std::runtime_error("GPUSplineManager: ndim must be 1-3");
    }

    GPUSplineTable spline;
    std::memset(&spline, 0, sizeof(GPUSplineTable));

    spline.ndim = ndim;
    spline.ncoeffs = static_cast<int>(coeffs.size());

    // Copy orders and compute strides
    // NOTE: orders[d] = B-spline degree (photospline convention)
    // naxes[d] = nknots[d] - degree - 1 = number of coefficients in dim d
    int stride = 1;
    for (int d = ndim - 1; d >= 0; --d) {
        spline.order[d] = orders[d];
        spline.nknots[d] = nknots[d];
        spline.strides[d] = stride;

        // Number of coefficients = nknots - degree - 1 (matching photospline's naxes)
        int naxes = nknots[d] - orders[d] - 1;
        stride *= naxes;

        // Compute extents from knots (matching photospline's searchcenters)
        // Valid range: [knots[degree], knots[naxes]]
        spline.extents_min[d] = knots[d][orders[d]];
        spline.extents_max[d] = knots[d][naxes];
    }

    // Allocate and copy knot vectors
    for (int d = 0; d < ndim; ++d) {
        size_t knot_bytes = nknots[d] * sizeof(double);
        CUDA_CHECK(cudaMalloc(&spline.knots[d], knot_bytes));
        CUDA_CHECK(cudaMemcpy(spline.knots[d], knots[d].data(), knot_bytes,
                              cudaMemcpyHostToDevice));
    }

    // Allocate and copy coefficients
    size_t coeff_bytes = coeffs.size() * sizeof(double);
    CUDA_CHECK(cudaMalloc(&spline.coeffs, coeff_bytes));
    CUDA_CHECK(cudaMemcpy(spline.coeffs, coeffs.data(), coeff_bytes,
                          cudaMemcpyHostToDevice));

    // Store host copy of metadata
    hostSplines_.push_back(spline);

    // Allocate device structure and copy
    GPUSplineTable* d_spline;
    CUDA_CHECK(cudaMalloc(&d_spline, sizeof(GPUSplineTable)));
    CUDA_CHECK(cudaMemcpy(d_spline, &spline, sizeof(GPUSplineTable),
                          cudaMemcpyHostToDevice));

    deviceSplines_.push_back(d_spline);

    return static_cast<int>(deviceSplines_.size() - 1);
}

const GPUSplineTable* GPUSplineManager::getDeviceSpline(int index) const {
    if (index < 0 || index >= static_cast<int>(deviceSplines_.size())) {
        return nullptr;
    }
    return deviceSplines_[index];
}

//==============================================================================
// Spline Evaluation Kernels
//==============================================================================

/**
 * @brief Kernel to evaluate a spline at multiple points
 */
__global__ void evaluateSplineAtPointsKernel(
    const GPUSplineTable* spline,
    const double* coords,  // [numPoints * ndim]
    double* results,       // [numPoints]
    int numPoints,
    int ndim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;

    if (ndim == 2) {
        double x = coords[tid * 2];
        double y = coords[tid * 2 + 1];
        results[tid] = evaluateSpline2D(*spline, x, y);
    } else if (ndim == 3) {
        double x = coords[tid * 3];
        double y = coords[tid * 3 + 1];
        double z = coords[tid * 3 + 2];
        results[tid] = evaluateSpline3D(*spline, x, y, z);
    }
}

/**
 * @brief Kernel to evaluate spline with derivative at multiple points
 */
__global__ void evaluateSplineWithDerivKernel(
    const GPUSplineTable* spline,
    const double* coords,  // [numPoints * 3]
    double* values,        // [numPoints]
    double* derivs,        // [numPoints]
    int derivDim,          // Which dimension to differentiate
    int numPoints
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;

    double x = coords[tid * 3];
    double y = coords[tid * 3 + 1];
    double z = coords[tid * 3 + 2];

    double val, deriv;
    evaluateSpline3DWithDerivative(*spline, x, y, z, derivDim, val, deriv);

    values[tid] = val;
    derivs[tid] = deriv;
}

//==============================================================================
// Wrapper Functions for Spline Evaluation
//==============================================================================

void evaluateSplineAtPoints(
    const GPUSplineTable* d_spline,
    const double* d_coords,
    double* d_results,
    int numPoints,
    int ndim,
    cudaStream_t stream
) {
    if (numPoints == 0) return;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    evaluateSplineAtPointsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_spline, d_coords, d_results, numPoints, ndim
    );

    CUDA_CHECK_KERNEL();
}

void evaluateSplineWithDerivatives(
    const GPUSplineTable* d_spline,
    const double* d_coords,
    double* d_values,
    double* d_derivs,
    int derivDim,
    int numPoints,
    cudaStream_t stream
) {
    if (numPoints == 0) return;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    evaluateSplineWithDerivKernel<<<gridSize, blockSize, 0, stream>>>(
        d_spline, d_coords, d_values, d_derivs, derivDim, numPoints
    );

    CUDA_CHECK_KERNEL();
}

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA
