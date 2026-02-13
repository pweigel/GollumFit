/**
 * @file GPUCommon.h
 * @brief Common CUDA utilities, error checking, and type definitions for GollumFit GPU acceleration.
 *
 * This file provides:
 * - CUDA error checking macros
 * - Common type definitions
 * - Memory management utilities
 * - Device property queries
 */

#ifndef GOLLUMFIT_GPU_COMMON_H
#define GOLLUMFIT_GPU_COMMON_H

#ifdef GOLLUMFIT_USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <memory>

namespace gollumfit {
namespace gpu {

//==============================================================================
// Error Checking Macros
//==============================================================================

/**
 * @brief CUDA error checking macro
 * Checks the return value of CUDA API calls and throws on error.
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

/**
 * @brief Check for CUDA errors after kernel launch (asynchronous check)
 */
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA kernel error at ") + __FILE__ + ":" +        \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

/**
 * @brief Synchronize and check for errors
 */
#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
        CUDA_CHECK_KERNEL();                                                   \
    } while (0)

//==============================================================================
// Constants
//==============================================================================

/// Number of fit parameters (matching FD<38> from PhysTools)
constexpr unsigned int NUM_FIT_PARAMS = 38;

/// Default block size for CUDA kernels
constexpr unsigned int DEFAULT_BLOCK_SIZE = 256;

/// Warp size (NVIDIA GPUs)
constexpr unsigned int WARP_SIZE = 32;

/// Maximum number of blocks for reduction kernels
constexpr unsigned int MAX_GRID_SIZE = 65535;

//==============================================================================
// Type Definitions
//==============================================================================

/// Use double precision for physics calculations (matches CPU code)
using real_t = double;

/// Index type for event arrays
using index_t = int32_t;

/// Size type for counts
using size_type = size_t;

//==============================================================================
// Memory Management
//==============================================================================

/**
 * @brief RAII wrapper for CUDA device memory
 */
template<typename T>
class DevicePtr {
public:
    DevicePtr() : ptr_(nullptr), size_(0) {}

    explicit DevicePtr(size_t count) : ptr_(nullptr), size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    ~DevicePtr() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Non-copyable
    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;

    // Movable
    DevicePtr(DevicePtr&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DevicePtr& operator=(DevicePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

    void copyFromHost(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyToHost(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void copyFromHostAsync(const T* host_data, size_t count, cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host_data, count * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
    }

    void copyToHostAsync(T* host_data, size_t count, cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(host_data, ptr_, count * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream));
    }

    void setZero() {
        if (ptr_ && size_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
        }
    }

    void setZeroAsync(cudaStream_t stream) {
        if (ptr_ && size_ > 0) {
            CUDA_CHECK(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream));
        }
    }

    void resize(size_t count) {
        if (count == size_) return;
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = count;
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

private:
    T* ptr_;
    size_t size_;
};

/**
 * @brief RAII wrapper for pinned host memory (for faster transfers)
 */
template<typename T>
class PinnedPtr {
public:
    PinnedPtr() : ptr_(nullptr), size_(0) {}

    explicit PinnedPtr(size_t count) : ptr_(nullptr), size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
        }
    }

    ~PinnedPtr() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }

    // Non-copyable
    PinnedPtr(const PinnedPtr&) = delete;
    PinnedPtr& operator=(const PinnedPtr&) = delete;

    // Movable
    PinnedPtr(PinnedPtr&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    PinnedPtr& operator=(PinnedPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T* ptr_;
    size_t size_;
};

//==============================================================================
// Device Properties
//==============================================================================

/**
 * @brief Query and store GPU device properties
 */
struct GPUDeviceInfo {
    int deviceId = -1;
    std::string name = "";
    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    size_t totalGlobalMem = 0;
    size_t sharedMemPerBlock = 0;
    int maxThreadsPerBlock = 0;
    int maxBlocksPerSM = 0;
    int multiProcessorCount = 0;
    int warpSize = 0;
    bool supportsDoublePrecision = false;
    bool supportsAtomicAddDouble = false;

    /// Default constructor - returns empty/invalid device info
    GPUDeviceInfo() = default;

    static GPUDeviceInfo query(int deviceId = 0) {
        GPUDeviceInfo info;
        info.deviceId = deviceId;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

        info.name = prop.name;
        info.computeCapabilityMajor = prop.major;
        info.computeCapabilityMinor = prop.minor;
        info.totalGlobalMem = prop.totalGlobalMem;
        info.sharedMemPerBlock = prop.sharedMemPerBlock;
        info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        info.maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
        info.multiProcessorCount = prop.multiProcessorCount;
        info.warpSize = prop.warpSize;

        // FP64 support check
        info.supportsDoublePrecision = (prop.major >= 2);

        // Native FP64 atomic add support (Ampere and later, SM >= 8.0)
        info.supportsAtomicAddDouble = (prop.major >= 8);

        return info;
    }

    void print() const {
        printf("GPU Device %d: %s\n", deviceId, name.c_str());
        printf("  Compute Capability: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);
        printf("  Total Global Memory: %.2f GB\n", totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %zu KB\n", sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", maxThreadsPerBlock);
        printf("  Multiprocessors: %d\n", multiProcessorCount);
        printf("  FP64 Support: %s\n", supportsDoublePrecision ? "Yes" : "No");
        printf("  Native FP64 Atomic Add: %s\n", supportsAtomicAddDouble ? "Yes" : "No");
    }
};

//==============================================================================
// Kernel Launch Helpers
//==============================================================================

/**
 * @brief Calculate optimal grid and block dimensions
 */
inline void calculateLaunchConfig(size_t numElements, int& gridSize, int& blockSize,
                                   int maxBlockSize = DEFAULT_BLOCK_SIZE) {
    blockSize = maxBlockSize;
    gridSize = static_cast<int>((numElements + blockSize - 1) / blockSize);

    // Clamp grid size to hardware limits
    if (gridSize > MAX_GRID_SIZE) {
        gridSize = MAX_GRID_SIZE;
    }
}

/**
 * @brief Calculate grid size for cooperative kernel launches
 */
inline int getMaxActiveBlocksPerSM(const void* kernel, int blockSize, size_t dynamicSharedMem = 0) {
    int numBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, kernel, blockSize, dynamicSharedMem));
    return numBlocks;
}

//==============================================================================
// Stream Management
//==============================================================================

/**
 * @brief RAII wrapper for CUDA streams
 */
class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    explicit CudaStream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    // Non-copyable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Movable
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    bool isComplete() const {
        cudaError_t result = cudaStreamQuery(stream_);
        if (result == cudaSuccess) return true;
        if (result == cudaErrorNotReady) return false;
        CUDA_CHECK(result);
        return false;
    }

private:
    cudaStream_t stream_;
};

//==============================================================================
// Atomic Operations (for pre-Ampere FP64 atomics compatibility)
//==============================================================================

// Device-only code: only compiled by nvcc, not by host C++ compiler
#ifdef __CUDACC__
/**
 * @brief Software emulation of atomicAdd for double on older GPUs
 * On SM 8.0+ (Ampere), native double atomics are used automatically.
 */
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    // Native double precision atomics available on SM 6.0+
    return atomicAdd(address, val);
#else
    // Software emulation using CAS for older architectures
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}
#endif // __CUDACC__

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA

#endif // GOLLUMFIT_GPU_COMMON_H
