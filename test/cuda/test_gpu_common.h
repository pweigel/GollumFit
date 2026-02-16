/**
 * @file test_gpu_common.h
 * @brief Common utilities for GPU tests.
 */

#ifndef TEST_GPU_COMMON_H
#define TEST_GPU_COMMON_H

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <random>

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double cpuTime = 0.0;
    double gpuTime = 0.0;
};

// Test suite class
class TestSuite {
public:
    void addResult(const TestResult& result) {
        results_.push_back(result);
        if (result.passed) {
            std::cout << "[PASS] " << result.name << std::endl;
        } else {
            std::cout << "[FAIL] " << result.name << ": " << result.message << std::endl;
        }
    }

    void printSummary() const {
        int passed = 0, failed = 0;
        for (const auto& r : results_) {
            if (r.passed) passed++;
            else failed++;
        }
        std::cout << "\n========================================\n";
        std::cout << "Test Summary: " << passed << " passed, " << failed << " failed\n";
        std::cout << "========================================\n";
    }

    bool allPassed() const {
        for (const auto& r : results_) {
            if (!r.passed) return false;
        }
        return true;
    }

private:
    std::vector<TestResult> results_;
};

// Comparison utilities
template<typename T>
bool compareValues(T cpu, T gpu, T tolerance, const std::string& name) {
    T diff = std::abs(cpu - gpu);
    T maxVal = std::max(std::abs(cpu), std::abs(gpu));
    T relErr = (maxVal > 0) ? diff / maxVal : diff;

    if (relErr > tolerance) {
        std::cerr << "  Mismatch in " << name << ": CPU=" << cpu
                  << ", GPU=" << gpu << ", relErr=" << relErr << std::endl;
        return false;
    }
    return true;
}

template<typename T>
bool compareArrays(const T* cpu, const T* gpu, size_t n, T tolerance, const std::string& name) {
    size_t mismatches = 0;
    T maxRelErr = 0;
    size_t maxErrIdx = 0;

    for (size_t i = 0; i < n; ++i) {
        T diff = std::abs(cpu[i] - gpu[i]);
        T maxVal = std::max(std::abs(cpu[i]), std::abs(gpu[i]));
        T relErr = (maxVal > 1e-15) ? diff / maxVal : diff;

        if (relErr > tolerance) {
            mismatches++;
            if (relErr > maxRelErr) {
                maxRelErr = relErr;
                maxErrIdx = i;
            }
        }
    }

    if (mismatches > 0) {
        std::cerr << "  Array " << name << ": " << mismatches << "/" << n
                  << " mismatches, max relErr=" << maxRelErr
                  << " at index " << maxErrIdx << std::endl;
        return false;
    }
    return true;
}

// Timer utility
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Random number generator for test data
class TestRNG {
public:
    TestRNG(unsigned int seed = 42) : gen_(seed), dist_(0.0, 1.0) {}

    double uniform() { return dist_(gen_); }
    double uniform(double min, double max) { return min + (max - min) * uniform(); }
    int uniformInt(int min, int max) { return min + static_cast<int>(uniform() * (max - min)); }

private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist_;
};

#endif // TEST_GPU_COMMON_H
