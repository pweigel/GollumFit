/**
 * @file test_gradient_verification.cu
 * @brief Verify GPU autodiff gradients against finite differences.
 *
 * This test computes gradients using both:
 * 1. GPU autodiff (forward-mode automatic differentiation)
 * 2. Finite differences (numerical approximation)
 *
 * And compares them to ensure autodiff is working correctly.
 */

#include "test_gpu_common.h"
#include <PhysTools/cuda/GPUAutodiff.h>
#include <PhysTools/autodiff.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <functional>

using namespace phys_tools::autodiff::gpu;
using CPUDual = phys_tools::autodiff::FD<38, double>;

//==============================================================================
// GPU Kernels for Gradient Testing
//==============================================================================

// Test function: f(x,y,z) = exp(x*y) + sin(z) + x^2*y + log(1+z^2)
__device__ double testFunction(double x, double y, double z) {
    return exp(x * y) + sin(z) + x * x * y + log(1.0 + z * z);
}

// Analytical gradients for verification
__device__ void testFunctionGradients(double x, double y, double z,
                                       double& df_dx, double& df_dy, double& df_dz) {
    df_dx = y * exp(x * y) + 2.0 * x * y;
    df_dy = x * exp(x * y) + x * x;
    df_dz = cos(z) + 2.0 * z / (1.0 + z * z);
}

__global__ void computeGradientAutodiffKernel(
    double x, double y, double z,
    double* value,
    double* gradients  // [3]: df/dx, df/dy, df/dz
) {
    if (threadIdx.x != 0) return;

    GPUDual<3> x_d(x, 0);
    GPUDual<3> y_d(y, 1);
    GPUDual<3> z_d(z, 2);

    // f(x,y,z) = exp(x*y) + sin(z) + x^2*y + log(1+z^2)
    GPUDual<3> f = exp(x_d * y_d) + sin(z_d) + x_d * x_d * y_d + log(GPUDual<3>(1.0) + z_d * z_d);

    *value = f.value();
    gradients[0] = f.derivative(0);
    gradients[1] = f.derivative(1);
    gradients[2] = f.derivative(2);
}

__global__ void computeGradientFiniteDiffKernel(
    double x, double y, double z,
    double epsilon,
    double* value,
    double* gradients  // [3]: df/dx, df/dy, df/dz (central differences)
) {
    if (threadIdx.x != 0) return;

    *value = testFunction(x, y, z);

    // Central differences: df/dx ≈ (f(x+ε) - f(x-ε)) / (2ε)
    gradients[0] = (testFunction(x + epsilon, y, z) - testFunction(x - epsilon, y, z)) / (2.0 * epsilon);
    gradients[1] = (testFunction(x, y + epsilon, z) - testFunction(x, y - epsilon, z)) / (2.0 * epsilon);
    gradients[2] = (testFunction(x, y, z + epsilon) - testFunction(x, y, z - epsilon)) / (2.0 * epsilon);
}

__global__ void computeAnalyticalGradientKernel(
    double x, double y, double z,
    double* value,
    double* gradients
) {
    if (threadIdx.x != 0) return;

    *value = testFunction(x, y, z);
    testFunctionGradients(x, y, z, gradients[0], gradients[1], gradients[2]);
}

//==============================================================================
// Test: Autodiff vs Analytical Gradient
//==============================================================================

TestResult testAutodiffVsAnalytical() {
    TestResult result;
    result.name = "Gradient Verification - Autodiff vs Analytical";

    // Test at multiple points
    std::vector<std::tuple<double, double, double>> testPoints = {
        {1.0, 2.0, 0.5},
        {0.5, 0.3, 1.0},
        {-0.5, 1.5, -0.3},
        {2.0, -1.0, 0.8},
        {0.1, 0.1, 0.1},
    };

    double* d_value;
    double* d_grads;
    cudaMalloc(&d_value, sizeof(double));
    cudaMalloc(&d_grads, 3 * sizeof(double));

    result.passed = true;
    double maxError = 0.0;

    for (const auto& point : testPoints) {
        double x = std::get<0>(point);
        double y = std::get<1>(point);
        double z = std::get<2>(point);

        // Autodiff
        computeGradientAutodiffKernel<<<1, 1>>>(x, y, z, d_value, d_grads);
        cudaDeviceSynchronize();

        double autodiffValue;
        double autodiffGrads[3];
        cudaMemcpy(&autodiffValue, d_value, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(autodiffGrads, d_grads, 3 * sizeof(double), cudaMemcpyDeviceToHost);

        // Analytical
        computeAnalyticalGradientKernel<<<1, 1>>>(x, y, z, d_value, d_grads);
        cudaDeviceSynchronize();

        double analyticalValue;
        double analyticalGrads[3];
        cudaMemcpy(&analyticalValue, d_value, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(analyticalGrads, d_grads, 3 * sizeof(double), cudaMemcpyDeviceToHost);

        // Compare values
        double valueError = std::abs(autodiffValue - analyticalValue);
        if (valueError > 1e-14) {
            result.passed = false;
            result.message = "Value mismatch";
        }

        // Compare gradients
        for (int i = 0; i < 3; ++i) {
            double gradError = std::abs(autodiffGrads[i] - analyticalGrads[i]);
            double relError = gradError / (std::abs(analyticalGrads[i]) + 1e-15);
            maxError = std::max(maxError, relError);

            if (relError > 1e-12) {
                result.passed = false;
                result.message = "Gradient mismatch at component " + std::to_string(i);
            }
        }
    }

    cudaFree(d_value);
    cudaFree(d_grads);

    std::cout << "  Max relative gradient error: " << maxError << std::endl;

    return result;
}

//==============================================================================
// Test: Autodiff vs Finite Differences
//==============================================================================

TestResult testAutodiffVsFiniteDiff() {
    TestResult result;
    result.name = "Gradient Verification - Autodiff vs Finite Differences";

    std::vector<std::tuple<double, double, double>> testPoints = {
        {1.0, 2.0, 0.5},
        {0.5, 0.3, 1.0},
        {-0.5, 1.5, -0.3},
        {2.0, -1.0, 0.8},
        {0.1, 0.1, 0.1},
        {3.0, 0.5, 2.0},
    };

    double epsilon = 1e-7;  // Step size for finite differences

    double* d_value;
    double* d_grads;
    cudaMalloc(&d_value, sizeof(double));
    cudaMalloc(&d_grads, 3 * sizeof(double));

    result.passed = true;
    double maxRelError = 0.0;

    for (const auto& point : testPoints) {
        double x = std::get<0>(point);
        double y = std::get<1>(point);
        double z = std::get<2>(point);

        // Autodiff
        computeGradientAutodiffKernel<<<1, 1>>>(x, y, z, d_value, d_grads);
        cudaDeviceSynchronize();

        double autodiffGrads[3];
        cudaMemcpy(autodiffGrads, d_grads, 3 * sizeof(double), cudaMemcpyDeviceToHost);

        // Finite differences
        computeGradientFiniteDiffKernel<<<1, 1>>>(x, y, z, epsilon, d_value, d_grads);
        cudaDeviceSynchronize();

        double finiteDiffGrads[3];
        cudaMemcpy(finiteDiffGrads, d_grads, 3 * sizeof(double), cudaMemcpyDeviceToHost);

        // Compare
        for (int i = 0; i < 3; ++i) {
            double absError = std::abs(autodiffGrads[i] - finiteDiffGrads[i]);
            double relError = absError / (std::abs(autodiffGrads[i]) + 1e-15);
            maxRelError = std::max(maxRelError, relError);

            // Finite differences have O(epsilon^2) error, so allow larger tolerance
            if (relError > 1e-5) {
                result.passed = false;
                result.message = "Large discrepancy at component " + std::to_string(i);
            }
        }
    }

    cudaFree(d_value);
    cudaFree(d_grads);

    std::cout << "  Max relative error vs finite diff: " << maxRelError << std::endl;

    return result;
}

//==============================================================================
// Test: Chain Rule Verification
//==============================================================================

// More complex function to test chain rule
__global__ void computeComplexGradientKernel(
    const double* params,  // [5] parameters
    double* value,
    double* gradients      // [5] gradients
) {
    if (threadIdx.x != 0) return;

    GPUDual<5> p[5];
    for (int i = 0; i < 5; ++i) {
        p[i] = GPUDual<5>(params[i], i);
    }

    // Complex function with multiple chain rule applications:
    // f = exp(p0 * sin(p1)) * log(1 + p2^2 + p3^2) / (1 + exp(-p4))
    GPUDual<5> term1 = exp(p[0] * sin(p[1]));
    GPUDual<5> term2 = log(GPUDual<5>(1.0) + p[2] * p[2] + p[3] * p[3]);
    GPUDual<5> term3 = GPUDual<5>(1.0) + exp(GPUDual<5>(0.0) - p[4]);
    GPUDual<5> f = term1 * term2 / term3;

    *value = f.value();
    for (int i = 0; i < 5; ++i) {
        gradients[i] = f.derivative(i);
    }
}

TestResult testChainRuleComplex() {
    TestResult result;
    result.name = "Gradient Verification - Complex Chain Rule";

    double params[5] = {0.5, 1.0, 0.3, 0.4, 0.2};
    double epsilon = 1e-7;

    double* d_params;
    double* d_value;
    double* d_grads;
    cudaMalloc(&d_params, 5 * sizeof(double));
    cudaMalloc(&d_value, sizeof(double));
    cudaMalloc(&d_grads, 5 * sizeof(double));

    // Autodiff
    cudaMemcpy(d_params, params, 5 * sizeof(double), cudaMemcpyHostToDevice);
    computeComplexGradientKernel<<<1, 1>>>(d_params, d_value, d_grads);
    cudaDeviceSynchronize();

    double autodiffValue;
    double autodiffGrads[5];
    cudaMemcpy(&autodiffValue, d_value, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(autodiffGrads, d_grads, 5 * sizeof(double), cudaMemcpyDeviceToHost);

    // Finite differences for each parameter
    double finiteDiffGrads[5];
    for (int i = 0; i < 5; ++i) {
        double paramsPlusEps[5];
        double paramsMinusEps[5];
        std::copy(params, params + 5, paramsPlusEps);
        std::copy(params, params + 5, paramsMinusEps);
        paramsPlusEps[i] += epsilon;
        paramsMinusEps[i] -= epsilon;

        cudaMemcpy(d_params, paramsPlusEps, 5 * sizeof(double), cudaMemcpyHostToDevice);
        computeComplexGradientKernel<<<1, 1>>>(d_params, d_value, d_grads);
        cudaDeviceSynchronize();
        double fPlus;
        cudaMemcpy(&fPlus, d_value, sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(d_params, paramsMinusEps, 5 * sizeof(double), cudaMemcpyHostToDevice);
        computeComplexGradientKernel<<<1, 1>>>(d_params, d_value, d_grads);
        cudaDeviceSynchronize();
        double fMinus;
        cudaMemcpy(&fMinus, d_value, sizeof(double), cudaMemcpyDeviceToHost);

        finiteDiffGrads[i] = (fPlus - fMinus) / (2.0 * epsilon);
    }

    cudaFree(d_params);
    cudaFree(d_value);
    cudaFree(d_grads);

    // Compare
    result.passed = true;
    double maxRelError = 0.0;

    std::cout << "  Parameter gradients:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        double relError = std::abs(autodiffGrads[i] - finiteDiffGrads[i]) /
                         (std::abs(autodiffGrads[i]) + 1e-15);
        maxRelError = std::max(maxRelError, relError);

        std::cout << "    p[" << i << "]: autodiff=" << autodiffGrads[i]
                  << ", finite_diff=" << finiteDiffGrads[i]
                  << ", rel_error=" << relError << std::endl;

        if (relError > 1e-5) {
            result.passed = false;
        }
    }

    if (!result.passed) {
        result.message = "Max relative error: " + std::to_string(maxRelError);
    }

    return result;
}

//==============================================================================
// Test: GPU vs CPU Autodiff Consistency
//==============================================================================

TestResult testGPUvsCPUAutodiff() {
    TestResult result;
    result.name = "Gradient Verification - GPU vs CPU Autodiff";

    TestRNG rng(98765);
    const int numTests = 20;

    result.passed = true;
    double maxRelError = 0.0;

    for (int t = 0; t < numTests; ++t) {
        double x = rng.uniform(-2.0, 2.0);
        double y = rng.uniform(-2.0, 2.0);
        double z = rng.uniform(-2.0, 2.0);

        // GPU autodiff
        double* d_value;
        double* d_grads;
        cudaMalloc(&d_value, sizeof(double));
        cudaMalloc(&d_grads, 3 * sizeof(double));

        computeGradientAutodiffKernel<<<1, 1>>>(x, y, z, d_value, d_grads);
        cudaDeviceSynchronize();

        double gpuValue;
        double gpuGrads[3];
        cudaMemcpy(&gpuValue, d_value, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpuGrads, d_grads, 3 * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_value);
        cudaFree(d_grads);

        // CPU autodiff using PhysTools FD
        CPUDual x_cpu(x, 0);
        CPUDual y_cpu(y, 1);
        CPUDual z_cpu(z, 2);

        // f(x,y,z) = exp(x*y) + sin(z) + x^2*y + log(1+z^2)
        CPUDual f_cpu = exp(x_cpu * y_cpu) + sin(z_cpu) + x_cpu * x_cpu * y_cpu +
                        log(CPUDual(1.0) + z_cpu * z_cpu);

        double cpuValue = f_cpu.value();
        double cpuGrads[3] = {f_cpu.derivative(0), f_cpu.derivative(1), f_cpu.derivative(2)};

        // Compare values
        double valueRelError = std::abs(gpuValue - cpuValue) / (std::abs(cpuValue) + 1e-15);
        if (valueRelError > 1e-14) {
            result.passed = false;
            result.message = "Value mismatch between GPU and CPU";
        }

        // Compare gradients
        for (int i = 0; i < 3; ++i) {
            double relError = std::abs(gpuGrads[i] - cpuGrads[i]) / (std::abs(cpuGrads[i]) + 1e-15);
            maxRelError = std::max(maxRelError, relError);

            if (relError > 1e-12) {
                result.passed = false;
            }
        }
    }

    std::cout << "  Max relative error GPU vs CPU: " << maxRelError << std::endl;

    if (!result.passed && result.message.empty()) {
        result.message = "Gradient mismatch: max error = " + std::to_string(maxRelError);
    }

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Gradient Verification Tests\n";
    std::cout << "========================================\n\n";

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

    suite.addResult(testAutodiffVsAnalytical());
    suite.addResult(testAutodiffVsFiniteDiff());
    suite.addResult(testChainRuleComplex());
    suite.addResult(testGPUvsCPUAutodiff());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
