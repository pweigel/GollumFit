/**
 * @file test_spline_evaluation.cu
 * @brief Tests for GPU B-spline evaluation.
 */

#include "test_gpu_common.h"
#include "cuda/GPUSplineTable.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace gollumfit::gpu;

// External function declarations (in gollumfit::gpu namespace)
namespace gollumfit {
namespace gpu {

extern void evaluateSplineAtPoints(
    const GPUSplineTable* d_spline,
    const double* d_coords,
    double* d_results,
    int numPoints,
    int ndim,
    cudaStream_t stream
);

extern void evaluateSplineWithDerivatives(
    const GPUSplineTable* d_spline,
    const double* d_coords,
    double* d_values,
    double* d_derivs,
    int derivDim,
    int numPoints,
    cudaStream_t stream
);

} // namespace gpu
} // namespace gollumfit

//==============================================================================
// CPU Reference Implementation (simplified B-spline)
//==============================================================================

/**
 * @brief Find knot span using binary search
 */
int cpuFindKnotSpan(double x, const std::vector<double>& knots, int nknots, int order) {
    int n = nknots - order - 1;  // naxes = number of basis functions
    if (x >= knots[n]) return n - 1;
    if (x <= knots[order]) return order;

    int low = order;
    int high = n;
    int mid = (low + high) / 2;

    while (x < knots[mid] || x >= knots[mid + 1]) {
        if (x < knots[mid]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

/**
 * @brief Evaluate B-spline basis functions using de Boor's algorithm
 */
void cpuEvaluateBasis(double x, const std::vector<double>& knots, int span, int order, std::vector<double>& basis) {
    std::vector<double> left(order + 1);
    std::vector<double> right(order + 1);

    basis[0] = 1.0;

    for (int j = 1; j <= order; ++j) {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        double saved = 0.0;

        for (int r = 0; r < j; ++r) {
            double temp = basis[r] / (right[r + 1] + left[j - r]);
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }
}

/**
 * @brief Evaluate 2D B-spline at given coordinates
 */
double cpuEvaluateSpline2D(
    const std::vector<std::vector<double>>& knots,
    const std::vector<double>& coeffs,
    const std::vector<int>& orders,
    const std::vector<int>& nknots,
    double x, double y
) {
    // Find spans
    int span0 = cpuFindKnotSpan(x, knots[0], nknots[0], orders[0]);
    int span1 = cpuFindKnotSpan(y, knots[1], nknots[1], orders[1]);

    // Evaluate basis functions (order+1 basis functions for degree=order)
    std::vector<double> basis0(orders[0] + 1);
    std::vector<double> basis1(orders[1] + 1);
    cpuEvaluateBasis(x, knots[0], span0, orders[0], basis0);
    cpuEvaluateBasis(y, knots[1], span1, orders[1], basis1);

    // Compute strides: naxes = nknots - order - 1
    int naxes1 = nknots[1] - orders[1] - 1;
    int stride0 = naxes1;
    int stride1 = 1;

    // Sum over tensor product (order+1 terms per dimension)
    double result = 0.0;
    for (int i = 0; i <= orders[0]; ++i) {
        int idx0 = span0 - orders[0] + i;
        for (int j = 0; j <= orders[1]; ++j) {
            int idx1 = span1 - orders[1] + j;
            int coeff_idx = idx0 * stride0 + idx1 * stride1;
            result += basis0[i] * basis1[j] * coeffs[coeff_idx];
        }
    }

    return result;
}

/**
 * @brief Evaluate 3D B-spline at given coordinates
 */
double cpuEvaluateSpline3D(
    const std::vector<std::vector<double>>& knots,
    const std::vector<double>& coeffs,
    const std::vector<int>& orders,
    const std::vector<int>& nknots,
    double x, double y, double z
) {
    // Find spans
    int span0 = cpuFindKnotSpan(x, knots[0], nknots[0], orders[0]);
    int span1 = cpuFindKnotSpan(y, knots[1], nknots[1], orders[1]);
    int span2 = cpuFindKnotSpan(z, knots[2], nknots[2], orders[2]);

    // Evaluate basis functions (order+1 basis functions for degree=order)
    std::vector<double> basis0(orders[0] + 1);
    std::vector<double> basis1(orders[1] + 1);
    std::vector<double> basis2(orders[2] + 1);
    cpuEvaluateBasis(x, knots[0], span0, orders[0], basis0);
    cpuEvaluateBasis(y, knots[1], span1, orders[1], basis1);
    cpuEvaluateBasis(z, knots[2], span2, orders[2], basis2);

    // Compute strides: naxes = nknots - order - 1
    int naxes1 = nknots[1] - orders[1] - 1;
    int naxes2 = nknots[2] - orders[2] - 1;
    int stride0 = naxes1 * naxes2;
    int stride1 = naxes2;
    int stride2 = 1;

    // Sum over tensor product (order+1 terms per dimension)
    double result = 0.0;
    for (int i = 0; i <= orders[0]; ++i) {
        int idx0 = span0 - orders[0] + i;
        for (int j = 0; j <= orders[1]; ++j) {
            int idx1 = span1 - orders[1] + j;
            for (int k = 0; k <= orders[2]; ++k) {
                int idx2 = span2 - orders[2] + k;
                int coeff_idx = idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
                result += basis0[i] * basis1[j] * basis2[k] * coeffs[coeff_idx];
            }
        }
    }

    return result;
}

//==============================================================================
// Test Data Generation
//==============================================================================

/**
 * @brief Create a simple 2D test spline (quadratic in both dimensions)
 * f(x,y) = x^2 + y^2 over [0,1]x[0,1]
 */
void createTestSpline2D(
    std::vector<std::vector<double>>& knots,
    std::vector<double>& coeffs,
    std::vector<int>& orders,
    std::vector<int>& nknots
) {
    // Order 3 (degree 3, cubic) in both dimensions
    orders = {3, 3};

    // Knot vectors: degree+1 = 4 repeated knots at each end for clamped spline
    knots.resize(2);
    knots[0] = {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0};
    knots[1] = {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0};

    nknots = {11, 11};

    // Number of coefficients (naxes): nknots - order - 1 = 7 in each dimension
    int naxes0 = 7;
    int naxes1 = 7;

    // Create coefficients for f(x,y) = x^2 + y^2
    // This is a rough approximation - in practice, splines are fit to data
    coeffs.resize(naxes0 * naxes1);

    // Simple coefficient pattern that approximates x^2 + y^2
    for (int i = 0; i < naxes0; ++i) {
        double xi = i / (double)(naxes0 - 1);
        for (int j = 0; j < naxes1; ++j) {
            double yj = j / (double)(naxes1 - 1);
            coeffs[i * naxes1 + j] = xi * xi + yj * yj;
        }
    }
}

/**
 * @brief Create a simple 3D test spline
 */
void createTestSpline3D(
    std::vector<std::vector<double>>& knots,
    std::vector<double>& coeffs,
    std::vector<int>& orders,
    std::vector<int>& nknots
) {
    // Order 3 (degree 3, cubic) in all dimensions
    orders = {3, 3, 3};

    // Knot vectors: degree+1 = 4 repeated knots at each end for clamped spline
    knots.resize(3);
    knots[0] = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0};
    knots[1] = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0};
    knots[2] = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0};

    nknots = {9, 9, 9};

    // Number of coefficients (naxes): nknots - order - 1 = 5 in each dimension
    int naxes0 = 5, naxes1 = 5, naxes2 = 5;

    // Create coefficients
    coeffs.resize(naxes0 * naxes1 * naxes2);

    for (int i = 0; i < naxes0; ++i) {
        double xi = i / (double)(naxes0 - 1);
        for (int j = 0; j < naxes1; ++j) {
            double yj = j / (double)(naxes1 - 1);
            for (int k = 0; k < naxes2; ++k) {
                double zk = k / (double)(naxes2 - 1);
                // f(x,y,z) = x + 2y + 3z
                coeffs[i * naxes1 * naxes2 + j * naxes2 + k] = xi + 2.0 * yj + 3.0 * zk;
            }
        }
    }
}

//==============================================================================
// Test Functions
//==============================================================================

TestResult testSpline2DEvaluation() {
    TestResult result;
    result.name = "Spline Evaluation - 2D Spline";

    // Create test spline
    std::vector<std::vector<double>> knots;
    std::vector<double> coeffs;
    std::vector<int> orders;
    std::vector<int> nknots;
    createTestSpline2D(knots, coeffs, orders, nknots);

    // Upload spline to GPU
    GPUSplineManager manager;
    int splineIdx = manager.uploadSpline(2, orders.data(), knots, coeffs, nknots.data());

    const GPUSplineTable* d_spline = manager.getDeviceSpline(splineIdx);

    // Test points
    TestRNG rng(42);
    const int numPoints = 100;
    std::vector<double> testX(numPoints);
    std::vector<double> testY(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        testX[i] = rng.uniform(0.1, 0.9);  // Avoid exact boundaries
        testY[i] = rng.uniform(0.1, 0.9);
    }

    // CPU evaluation
    std::vector<double> cpuResults(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        cpuResults[i] = cpuEvaluateSpline2D(knots, coeffs, orders, nknots, testX[i], testY[i]);
    }

    // GPU evaluation
    std::vector<double> coords(numPoints * 2);
    for (int i = 0; i < numPoints; ++i) {
        coords[i * 2] = testX[i];
        coords[i * 2 + 1] = testY[i];
    }

    double* d_coords;
    double* d_results;
    cudaMalloc(&d_coords, numPoints * 2 * sizeof(double));
    cudaMalloc(&d_results, numPoints * sizeof(double));

    cudaMemcpy(d_coords, coords.data(), numPoints * 2 * sizeof(double), cudaMemcpyHostToDevice);

    // Call GPU evaluation
    evaluateSplineAtPoints(d_spline, d_coords, d_results, numPoints, 2, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> gpuResults(numPoints);
    cudaMemcpy(gpuResults.data(), d_results, numPoints * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_coords);
    cudaFree(d_results);

    // Compare results
    const double tol = 1e-10;
    result.passed = compareArrays(cpuResults.data(), gpuResults.data(), numPoints, tol, "spline values");

    if (!result.passed) {
        result.message = "2D spline evaluation mismatch";
    }

    return result;
}

TestResult testSpline3DEvaluation() {
    TestResult result;
    result.name = "Spline Evaluation - 3D Spline";

    // Create test spline
    std::vector<std::vector<double>> knots;
    std::vector<double> coeffs;
    std::vector<int> orders;
    std::vector<int> nknots;
    createTestSpline3D(knots, coeffs, orders, nknots);

    // Upload spline to GPU
    GPUSplineManager manager;
    int splineIdx = manager.uploadSpline(3, orders.data(), knots, coeffs, nknots.data());

    const GPUSplineTable* d_spline = manager.getDeviceSpline(splineIdx);

    // Test points
    TestRNG rng(123);
    const int numPoints = 100;
    std::vector<double> testX(numPoints);
    std::vector<double> testY(numPoints);
    std::vector<double> testZ(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        testX[i] = rng.uniform(0.1, 0.9);
        testY[i] = rng.uniform(0.1, 0.9);
        testZ[i] = rng.uniform(0.1, 0.9);
    }

    // CPU evaluation
    std::vector<double> cpuResults(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        cpuResults[i] = cpuEvaluateSpline3D(knots, coeffs, orders, nknots, testX[i], testY[i], testZ[i]);
    }

    // GPU evaluation
    std::vector<double> coords(numPoints * 3);
    for (int i = 0; i < numPoints; ++i) {
        coords[i * 3] = testX[i];
        coords[i * 3 + 1] = testY[i];
        coords[i * 3 + 2] = testZ[i];
    }

    double* d_coords;
    double* d_results;
    cudaMalloc(&d_coords, numPoints * 3 * sizeof(double));
    cudaMalloc(&d_results, numPoints * sizeof(double));

    cudaMemcpy(d_coords, coords.data(), numPoints * 3 * sizeof(double), cudaMemcpyHostToDevice);

    // Call GPU evaluation
    evaluateSplineAtPoints(d_spline, d_coords, d_results, numPoints, 3, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> gpuResults(numPoints);
    cudaMemcpy(gpuResults.data(), d_results, numPoints * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_coords);
    cudaFree(d_results);

    // Compare results
    const double tol = 1e-10;
    result.passed = compareArrays(cpuResults.data(), gpuResults.data(), numPoints, tol, "spline values");

    if (!result.passed) {
        result.message = "3D spline evaluation mismatch";
    }

    return result;
}

TestResult testSplineManager() {
    TestResult result;
    result.name = "Spline Manager - Upload and Retrieval";

    // Create multiple splines
    std::vector<std::vector<double>> knots2D, knots3D;
    std::vector<double> coeffs2D, coeffs3D;
    std::vector<int> orders2D, orders3D;
    std::vector<int> nknots2D, nknots3D;

    createTestSpline2D(knots2D, coeffs2D, orders2D, nknots2D);
    createTestSpline3D(knots3D, coeffs3D, orders3D, nknots3D);

    GPUSplineManager manager;

    // Upload multiple splines
    int idx1 = manager.uploadSpline(2, orders2D.data(), knots2D, coeffs2D, nknots2D.data());
    int idx2 = manager.uploadSpline(3, orders3D.data(), knots3D, coeffs3D, nknots3D.data());

    result.passed = true;

    // Verify indices
    if (idx1 != 0 || idx2 != 1) {
        result.passed = false;
        result.message = "Spline indices not sequential";
    }

    // Verify retrieval
    const GPUSplineTable* s1 = manager.getDeviceSpline(0);
    const GPUSplineTable* s2 = manager.getDeviceSpline(1);
    const GPUSplineTable* s3 = manager.getDeviceSpline(2);  // Should be null

    if (s1 == nullptr || s2 == nullptr) {
        result.passed = false;
        result.message = "Valid splines returned nullptr";
    }

    if (s3 != nullptr) {
        result.passed = false;
        result.message = "Invalid index did not return nullptr";
    }

    // Verify count
    if (manager.getNumSplines() != 2) {
        result.passed = false;
        result.message = "Spline count incorrect";
    }

    return result;
}

TestResult testSplineBoundsChecking() {
    TestResult result;
    result.name = "Spline Evaluation - Bounds Checking";

    // Create test spline
    std::vector<std::vector<double>> knots;
    std::vector<double> coeffs;
    std::vector<int> orders;
    std::vector<int> nknots;
    createTestSpline2D(knots, coeffs, orders, nknots);

    GPUSplineManager manager;
    int splineIdx = manager.uploadSpline(2, orders.data(), knots, coeffs, nknots.data());
    const GPUSplineTable* d_spline = manager.getDeviceSpline(splineIdx);

    // Test out-of-bounds points
    std::vector<double> coords = {
        -0.5, 0.5,    // x out of bounds (low)
        1.5, 0.5,     // x out of bounds (high)
        0.5, -0.5,    // y out of bounds (low)
        0.5, 1.5,     // y out of bounds (high)
        0.5, 0.5      // valid point
    };

    const int numPoints = 5;
    double* d_coords;
    double* d_results;
    cudaMalloc(&d_coords, numPoints * 2 * sizeof(double));
    cudaMalloc(&d_results, numPoints * sizeof(double));

    cudaMemcpy(d_coords, coords.data(), numPoints * 2 * sizeof(double), cudaMemcpyHostToDevice);

    evaluateSplineAtPoints(d_spline, d_coords, d_results, numPoints, 2, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> gpuResults(numPoints);
    cudaMemcpy(gpuResults.data(), d_results, numPoints * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_coords);
    cudaFree(d_results);

    // Check that out-of-bounds return 0
    result.passed = true;
    for (int i = 0; i < 4; ++i) {
        if (gpuResults[i] != 0.0) {
            result.passed = false;
            result.message = "Out-of-bounds point did not return 0";
            std::cerr << "  Point " << i << " returned " << gpuResults[i] << " (expected 0)\n";
        }
    }

    // Check that valid point returns non-zero
    if (gpuResults[4] == 0.0) {
        result.passed = false;
        result.message = "Valid point incorrectly returned 0";
    }

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "B-Spline Evaluation Tests\n";
    std::cout << "========================================\n\n";

    // Check for CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n\n";

    TestSuite suite;

    suite.addResult(testSpline2DEvaluation());
    suite.addResult(testSpline3DEvaluation());
    suite.addResult(testSplineManager());
    suite.addResult(testSplineBoundsChecking());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
