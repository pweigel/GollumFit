# CMakeLists.txt for GollumFit CUDA tests

cmake_minimum_required(VERSION 3.18)

# This should be included from the main GollumFit CMakeLists.txt
# with USE_CUDA=ON

if(NOT USE_CUDA)
    message(STATUS "CUDA tests skipped (USE_CUDA=OFF)")
    return()
endif()

# Test common header
set(TEST_COMMON_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_common.h)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../PhysTools
    ${CMAKE_CURRENT_SOURCE_DIR}
)

#==============================================================================
# Test executables
#==============================================================================

# GPU Autodiff Test
add_executable(test_gpu_autodiff
    test_gpu_autodiff.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_gpu_autodiff
    GollumFit
    CUDA::cudart
)
set_target_properties(test_gpu_autodiff PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Histogram Reduction Test
add_executable(test_histogram_reduction
    test_histogram_reduction.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/histogram_reduction.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_histogram_reduction
    CUDA::cudart
)
target_compile_definitions(test_histogram_reduction PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(test_histogram_reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Likelihood Test
add_executable(test_likelihood
    test_likelihood.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/likelihood.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_likelihood
    CUDA::cudart
)
target_compile_definitions(test_likelihood PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(test_likelihood PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Spline Evaluation Test
add_executable(test_spline_evaluation
    test_spline_evaluation.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/GPUSplineTable.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_spline_evaluation
    CUDA::cudart
)
target_compile_definitions(test_spline_evaluation PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(test_spline_evaluation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Integration Test (requires full GollumFit library)
add_executable(test_integration
    test_integration.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_integration
    GollumFit
    CUDA::cudart
)
set_target_properties(test_integration PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Numerical Precision Test
add_executable(test_numerical_precision
    test_numerical_precision.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/histogram_reduction.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/likelihood.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_numerical_precision
    CUDA::cudart
)
target_compile_definitions(test_numerical_precision PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(test_numerical_precision PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Gradient Verification Test
add_executable(test_gradient_verification
    test_gradient_verification.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_gradient_verification
    GollumFit
    CUDA::cudart
)
set_target_properties(test_gradient_verification PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Adjoint Gradient Validation Test
add_executable(test_adjoint_gradient
    test_adjoint_gradient.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_adjoint_gradient
    GollumFit
    CUDA::cudart
)
set_target_properties(test_adjoint_gradient PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Reference Spline Caching Test
add_executable(test_reference_caching
    test_reference_caching.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(test_reference_caching
    GollumFit
    CUDA::cudart
)
set_target_properties(test_reference_caching PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#==============================================================================
# Benchmark executables
#==============================================================================

# Histogram Benchmark
add_executable(benchmark_histogram
    benchmark_histogram.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/histogram_reduction.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_histogram
    CUDA::cudart
)
target_compile_definitions(benchmark_histogram PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(benchmark_histogram PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Likelihood Benchmark
add_executable(benchmark_likelihood
    benchmark_likelihood.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/histogram_reduction.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/likelihood.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_likelihood
    CUDA::cudart
)
target_compile_definitions(benchmark_likelihood PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(benchmark_likelihood PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Weighting Benchmark
add_executable(benchmark_weighting
    benchmark_weighting.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/event_weighting.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_weighting
    GollumFit
    CUDA::cudart
)
set_target_properties(benchmark_weighting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Full Fit Iteration Benchmark (realistic minimization simulation)
add_executable(benchmark_fit_iteration
    benchmark_fit_iteration.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/event_weighting.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/histogram_reduction.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/kernels/likelihood.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_fit_iteration
    GollumFit
    CUDA::cudart
)
set_target_properties(benchmark_fit_iteration PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Reweight Component Analysis Benchmark
add_executable(benchmark_reweight_components
    benchmark_reweight_components.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_reweight_components
    GollumFit
    CUDA::cudart
)
target_compile_definitions(benchmark_reweight_components PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(benchmark_reweight_components PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# pow() Optimization Benchmark
add_executable(benchmark_pow_optimization
    benchmark_pow_optimization.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_pow_optimization
    GollumFit
    CUDA::cudart
)
target_compile_definitions(benchmark_pow_optimization PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(benchmark_pow_optimization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Conv Flux Optimization Benchmark
add_executable(benchmark_conv_flux_optimization
    benchmark_conv_flux_optimization.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_conv_flux_optimization
    GollumFit
    CUDA::cudart
)
target_compile_definitions(benchmark_conv_flux_optimization PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(benchmark_conv_flux_optimization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Mixed Precision Optimization Benchmark
add_executable(benchmark_mixed_precision
    benchmark_mixed_precision.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_mixed_precision
    GollumFit
    CUDA::cudart
)
target_compile_definitions(benchmark_mixed_precision PRIVATE GOLLUMFIT_USE_CUDA)
set_target_properties(benchmark_mixed_precision PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Gradient Scaling Benchmark
add_executable(benchmark_gradient_scaling
    benchmark_gradient_scaling.cu
    ${TEST_COMMON_HEADER}
)
target_link_libraries(benchmark_gradient_scaling
    GollumFit
    CUDA::cudart
)
set_target_properties(benchmark_gradient_scaling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#==============================================================================
# CTest integration
#==============================================================================

enable_testing()

add_test(NAME GPUAutodiffTest COMMAND test_gpu_autodiff)
add_test(NAME HistogramReductionTest COMMAND test_histogram_reduction)
add_test(NAME LikelihoodTest COMMAND test_likelihood)
add_test(NAME SplineEvaluationTest COMMAND test_spline_evaluation)
add_test(NAME IntegrationTest COMMAND test_integration)
add_test(NAME NumericalPrecisionTest COMMAND test_numerical_precision)
add_test(NAME GradientVerificationTest COMMAND test_gradient_verification)
add_test(NAME AdjointGradientTest COMMAND test_adjoint_gradient)
add_test(NAME ReferenceCachingTest COMMAND test_reference_caching)

# Set test properties
set_tests_properties(
    GPUAutodiffTest
    HistogramReductionTest
    LikelihoodTest
    SplineEvaluationTest
    IntegrationTest
    NumericalPrecisionTest
    GradientVerificationTest
    AdjointGradientTest
    ReferenceCachingTest
    PROPERTIES
    LABELS "GPU"
    TIMEOUT 300
)

#==============================================================================
# Custom target to run all GPU tests
#==============================================================================

add_custom_target(run_gpu_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L GPU --output-on-failure
    DEPENDS test_gpu_autodiff test_histogram_reduction test_likelihood test_spline_evaluation test_integration test_numerical_precision test_gradient_verification test_reference_caching
    COMMENT "Running all GPU tests"
)

#==============================================================================
# Custom target to run all benchmarks
#==============================================================================

add_custom_target(run_benchmarks
    COMMAND benchmark_histogram
    COMMAND benchmark_likelihood
    COMMAND benchmark_weighting
    COMMAND benchmark_fit_iteration
    COMMAND benchmark_reweight_components
    COMMAND benchmark_conv_flux_optimization
    DEPENDS benchmark_histogram benchmark_likelihood benchmark_weighting benchmark_fit_iteration benchmark_reweight_components benchmark_conv_flux_optimization
    COMMENT "Running all GPU benchmarks"
)

message(STATUS "CUDA tests configured:")
message(STATUS "  - test_gpu_autodiff")
message(STATUS "  - test_histogram_reduction")
message(STATUS "  - test_likelihood")
message(STATUS "  - test_spline_evaluation")
message(STATUS "  - test_integration")
message(STATUS "  - test_numerical_precision")
message(STATUS "  - test_gradient_verification")
message(STATUS "  - test_reference_caching")
message(STATUS "CUDA benchmarks configured:")
message(STATUS "  - benchmark_histogram")
message(STATUS "  - benchmark_likelihood")
message(STATUS "  - benchmark_weighting")
message(STATUS "  - benchmark_fit_iteration")
message(STATUS "  - benchmark_reweight_components")
message(STATUS "  - benchmark_pow_optimization")
message(STATUS "  - benchmark_conv_flux_optimization")
