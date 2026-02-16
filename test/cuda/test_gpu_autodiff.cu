/**
 * @file test_gpu_autodiff.cu
 * @brief Tests for GPU autodiff implementation comparing against CPU FD<38>.
 */

#include "test_gpu_common.h"
#include <PhysTools/cuda/GPUAutodiff.h>
#include <PhysTools/autodiff.h>
#include <cuda_runtime.h>

// GPU autodiff is in phys_tools::autodiff::gpu namespace
using namespace phys_tools::autodiff::gpu;  // For GPUDual
using CPUDual = phys_tools::autodiff::FD<38, double>;

//==============================================================================
// GPU Kernel for Autodiff Tests
//==============================================================================

__global__ void testBasicArithmeticKernel(
    double* results,      // [8] values
    double* gradients,    // [8 * 38] gradients
    double a_val, double b_val,
    int a_idx, int b_idx
) {
    if (threadIdx.x != 0) return;

    // Create dual numbers (using constructors, not static methods)
    GPUDual<38> a(a_val, a_idx);  // variable: value and index
    GPUDual<38> b(b_val, b_idx);  // variable: value and index
    GPUDual<38> c(2.5);           // constant: just value

    // Test operations
    GPUDual<38> r0 = a + b;           // addition
    GPUDual<38> r1 = a - b;           // subtraction
    GPUDual<38> r2 = a * b;           // multiplication
    GPUDual<38> r3 = a / b;           // division
    GPUDual<38> r4 = a + c;           // add constant
    GPUDual<38> r5 = a * c;           // multiply constant
    GPUDual<38> r6 = (a + b) * c;     // compound
    GPUDual<38> r7 = a * a + b * b;   // quadratic

    // Store results
    results[0] = r0.value();
    results[1] = r1.value();
    results[2] = r2.value();
    results[3] = r3.value();
    results[4] = r4.value();
    results[5] = r5.value();
    results[6] = r6.value();
    results[7] = r7.value();

    // Store gradients
    for (int i = 0; i < 38; ++i) {
        gradients[0 * 38 + i] = r0.derivative(i);
        gradients[1 * 38 + i] = r1.derivative(i);
        gradients[2 * 38 + i] = r2.derivative(i);
        gradients[3 * 38 + i] = r3.derivative(i);
        gradients[4 * 38 + i] = r4.derivative(i);
        gradients[5 * 38 + i] = r5.derivative(i);
        gradients[6 * 38 + i] = r6.derivative(i);
        gradients[7 * 38 + i] = r7.derivative(i);
    }
}

__global__ void testMathFunctionsKernel(
    double* results,      // [6] values
    double* gradients,    // [6 * 38] gradients
    double x_val,
    int x_idx
) {
    if (threadIdx.x != 0) return;

    GPUDual<38> x(x_val, x_idx);  // variable: value and index

    // Test math functions
    GPUDual<38> r0 = exp(x);
    GPUDual<38> r1 = log(x);
    GPUDual<38> r2 = sqrt(x);
    GPUDual<38> r3 = pow(x, 2.5);
    GPUDual<38> r4 = sin(x);
    GPUDual<38> r5 = cos(x);

    // Store results
    results[0] = r0.value();
    results[1] = r1.value();
    results[2] = r2.value();
    results[3] = r3.value();
    results[4] = r4.value();
    results[5] = r5.value();

    // Store gradients
    for (int i = 0; i < 38; ++i) {
        gradients[0 * 38 + i] = r0.derivative(i);
        gradients[1 * 38 + i] = r1.derivative(i);
        gradients[2 * 38 + i] = r2.derivative(i);
        gradients[3 * 38 + i] = r3.derivative(i);
        gradients[4 * 38 + i] = r4.derivative(i);
        gradients[5 * 38 + i] = r5.derivative(i);
    }
}

//==============================================================================
// Test Functions
//==============================================================================

TestResult testBasicArithmetic() {
    TestResult result;
    result.name = "GPU Autodiff - Basic Arithmetic";

    const double a_val = 3.0;
    const double b_val = 2.0;
    const int a_idx = 5;
    const int b_idx = 10;

    // CPU computation using PhysTools autodiff
    // Use constructor: FD(value, index) for variables, FD(value) for constants
    CPUDual a_cpu(a_val, a_idx);  // variable with gradient 1 at index a_idx
    CPUDual b_cpu(b_val, b_idx);  // variable with gradient 1 at index b_idx
    CPUDual c_cpu(2.5);           // constant (zero gradient)

    std::vector<CPUDual> cpu_results(8);
    cpu_results[0] = a_cpu + b_cpu;
    cpu_results[1] = a_cpu - b_cpu;
    cpu_results[2] = a_cpu * b_cpu;
    cpu_results[3] = a_cpu / b_cpu;
    cpu_results[4] = a_cpu + c_cpu;
    cpu_results[5] = a_cpu * c_cpu;
    cpu_results[6] = (a_cpu + b_cpu) * c_cpu;
    cpu_results[7] = a_cpu * a_cpu + b_cpu * b_cpu;

    // GPU computation
    double* d_results;
    double* d_gradients;
    cudaMalloc(&d_results, 8 * sizeof(double));
    cudaMalloc(&d_gradients, 8 * 38 * sizeof(double));

    testBasicArithmeticKernel<<<1, 1>>>(d_results, d_gradients, a_val, b_val, a_idx, b_idx);
    cudaDeviceSynchronize();

    std::vector<double> h_results(8);
    std::vector<double> h_gradients(8 * 38);
    cudaMemcpy(h_results.data(), d_results, 8 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gradients.data(), d_gradients, 8 * 38 * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_results);
    cudaFree(d_gradients);

    // Compare results
    const double tol = 1e-12;
    result.passed = true;

    const char* opNames[] = {"a+b", "a-b", "a*b", "a/b", "a+c", "a*c", "(a+b)*c", "a^2+b^2"};
    for (int i = 0; i < 8; ++i) {
        if (!compareValues(cpu_results[i].value(), h_results[i], tol, opNames[i])) {
            result.passed = false;
            result.message = std::string("Value mismatch in ") + opNames[i];
        }
        for (int j = 0; j < 38; ++j) {
            if (!compareValues(cpu_results[i].derivative(j), h_gradients[i * 38 + j], tol,
                              std::string(opNames[i]) + " grad[" + std::to_string(j) + "]")) {
                result.passed = false;
                result.message = std::string("Gradient mismatch in ") + opNames[i];
            }
        }
    }

    return result;
}

TestResult testMathFunctions() {
    TestResult result;
    result.name = "GPU Autodiff - Math Functions";

    const double x_val = 1.5;
    const int x_idx = 3;

    // CPU computation
    CPUDual x_cpu(x_val, x_idx);  // variable with gradient 1 at x_idx

    std::vector<CPUDual> cpu_results(6);
    cpu_results[0] = exp(x_cpu);
    cpu_results[1] = log(x_cpu);
    cpu_results[2] = sqrt(x_cpu);
    cpu_results[3] = pow(x_cpu, 2.5);
    cpu_results[4] = sin(x_cpu);
    cpu_results[5] = cos(x_cpu);

    // GPU computation
    double* d_results;
    double* d_gradients;
    cudaMalloc(&d_results, 6 * sizeof(double));
    cudaMalloc(&d_gradients, 6 * 38 * sizeof(double));

    testMathFunctionsKernel<<<1, 1>>>(d_results, d_gradients, x_val, x_idx);
    cudaDeviceSynchronize();

    std::vector<double> h_results(6);
    std::vector<double> h_gradients(6 * 38);
    cudaMemcpy(h_results.data(), d_results, 6 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gradients.data(), d_gradients, 6 * 38 * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_results);
    cudaFree(d_gradients);

    // Compare results
    const double tol = 1e-10;
    result.passed = true;

    const char* funcNames[] = {"exp", "log", "sqrt", "pow", "sin", "cos"};
    for (int i = 0; i < 6; ++i) {
        if (!compareValues(cpu_results[i].value(), h_results[i], tol, funcNames[i])) {
            result.passed = false;
            result.message = std::string("Value mismatch in ") + funcNames[i];
        }
        // Check the active gradient component
        if (!compareValues(cpu_results[i].derivative(x_idx), h_gradients[i * 38 + x_idx], tol,
                          std::string(funcNames[i]) + " grad")) {
            result.passed = false;
            result.message = std::string("Gradient mismatch in ") + funcNames[i];
        }
    }

    return result;
}

TestResult testChainRule() {
    TestResult result;
    result.name = "GPU Autodiff - Chain Rule (compound expressions)";

    // Test f(x,y) = exp(x*y) + log(x+y) at x=2, y=3
    // df/dx = y*exp(x*y) + 1/(x+y)
    // df/dy = x*exp(x*y) + 1/(x+y)

    const double x_val = 2.0;
    const double y_val = 3.0;

    // Expected values
    double expected_val = std::exp(x_val * y_val) + std::log(x_val + y_val);
    double expected_dx = y_val * std::exp(x_val * y_val) + 1.0 / (x_val + y_val);
    double expected_dy = x_val * std::exp(x_val * y_val) + 1.0 / (x_val + y_val);

    // CPU computation
    CPUDual x_cpu(x_val, 0);  // variable at index 0
    CPUDual y_cpu(y_val, 1);  // variable at index 1

    CPUDual f_cpu = exp(x_cpu * y_cpu) + log(x_cpu + y_cpu);

    // Verify CPU matches analytical
    const double tol = 1e-10;
    result.passed = true;

    if (!compareValues(expected_val, f_cpu.value(), tol, "f value")) {
        result.passed = false;
    }
    if (!compareValues(expected_dx, f_cpu.derivative(0), tol, "df/dx")) {
        result.passed = false;
    }
    if (!compareValues(expected_dy, f_cpu.derivative(1), tol, "df/dy")) {
        result.passed = false;
    }

    if (!result.passed) {
        result.message = "Chain rule test failed";
    }

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "GPU Autodiff Tests\n";
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

    suite.addResult(testBasicArithmetic());
    suite.addResult(testMathFunctions());
    suite.addResult(testChainRule());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
