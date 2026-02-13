/**
 * @file GPUEventDataPacked.h
 * @brief Packed event data structures for optimized GPU memory access.
 *
 * This file provides alternative memory layouts that group frequently
 * co-accessed data together for better cache utilization and vectorized loads.
 *
 * Optimization target: Conv Flux Assembly which loads 19 arrays per event
 * - cachedConvWeight (1)
 * - hadronic corrections (10)
 * - cosmic ray corrections (6)
 * - atmospheric corrections (2)
 *
 * By packing these into contiguous structures, we:
 * 1. Improve L2 cache hit rate (one cache line loads multiple values)
 * 2. Enable vectorized loads (double2, double4)
 * 3. Reduce total number of memory transactions
 */

#ifndef GOLLUMFIT_GPU_EVENT_DATA_PACKED_H
#define GOLLUMFIT_GPU_EVENT_DATA_PACKED_H

#ifdef GOLLUMFIT_USE_CUDA

#include "GPUCommon.h"
#include "GPUEventData.h"
#include <cuda_runtime.h>

namespace gollumfit {
namespace gpu {

//==============================================================================
// Packed Data Structures for Optimized Memory Access
//==============================================================================

/**
 * @brief Packed hadronic corrections (10 doubles = 80 bytes)
 *
 * Aligned to 16 bytes for optimal memory access.
 * Can be loaded as 5x double2 or 2x double4 + 1x double2.
 */
struct __align__(16) PackedHadronic {
    double HEkp;        // P_HEKP
    double HEkm;        // P_HEKM
    double VHE1pip;     // P_VHE1PIP
    double VHE1pim;     // P_VHE1PIM
    double VHE3kp;      // P_VHE3KP
    double VHE3km;      // P_VHE3KM
    double VHE3pip;     // P_VHE3PIP
    double VHE3pim;     // P_VHE3PIM
    double VHE3p;       // P_VHE3P
    double VHE3n;       // P_VHE3N
};

/**
 * @brief Packed cosmic ray corrections (6 doubles = 48 bytes)
 *
 * Aligned to 16 bytes. Can be loaded as 3x double2.
 */
struct __align__(16) PackedCosmicRay {
    double cr1;     // P_CR1
    double cr2;     // P_CR2
    double cr3;     // P_CR3
    double cr4;     // P_CR4
    double cr5;     // P_CR5
    double cr6;     // P_CR6
};

/**
 * @brief Packed atmospheric corrections (2 doubles = 16 bytes)
 *
 * Naturally aligned, can be loaded as 1x double2.
 */
struct __align__(16) PackedAtmospheric {
    double atmDensity;   // P_ADU weight
    double kaonLosses;   // P_KLU weight
};

/**
 * @brief Packed ice gradient corrections (9 doubles = 72 bytes, padded to 80)
 *
 * Padded to allow 5x double2 loads.
 */
struct __align__(16) PackedIceGrad {
    double grad0;
    double grad1;
    double grad2;
    double grad3;
    double grad4;
    double grad5;
    double grad6;
    double grad7;
    double grad8;
    double _pad;    // Padding for alignment
};

/**
 * @brief Combined conv flux data for a single event (152 bytes)
 *
 * Groups all data needed for conv flux calculation into a single
 * contiguous structure. This allows one memory transaction to load
 * all related data instead of 19 separate loads.
 */
struct __align__(16) PackedConvFluxData {
    double convWeight;          // Base conventional flux weight
    PackedHadronic hadronic;    // 10 hadronic corrections
    PackedCosmicRay cosmicRay;  // 6 cosmic ray corrections
    PackedAtmospheric atm;      // 2 atmospheric corrections
};

/**
 * @brief Full packed event data for weight calculation
 *
 * Groups all data needed for a complete weight calculation.
 * Total size: ~256 bytes (power of 2 for optimal alignment)
 */
struct __align__(256) PackedEventData {
    // Primary physics (32 bytes)
    float energy;
    float zenith;
    float primaryEnergy;
    float primaryZenith;
    int32_t primaryType;
    int32_t numEventsInBin;
    int32_t binIndex;
    int32_t _pad1;

    // Flux weights (24 bytes)
    double convWeight;
    double promptWeight;
    double astroWeight;

    // Hadronic (80 bytes)
    PackedHadronic hadronic;

    // Cosmic ray (48 bytes)
    PackedCosmicRay cosmicRay;

    // Atmospheric (16 bytes)
    PackedAtmospheric atm;

    // Ice gradients (80 bytes with padding)
    PackedIceGrad iceGrad;
};

//==============================================================================
// Packed Parameter Structure
//==============================================================================

/**
 * @brief Packed fit parameters for efficient shared memory broadcast
 *
 * Groups parameters by their usage pattern in the weight calculation.
 */
struct __align__(16) PackedParams {
    // Normalizations (16 bytes)
    double convNorm;
    double promptNorm;

    // Atmospheric (16 bytes)
    double adu;
    double klu;

    // Hadronic (80 bytes) - matches PackedHadronic order
    double hekp, hekm;
    double vhe1pip, vhe1pim;
    double vhe3kp, vhe3km;
    double vhe3pip, vhe3pim;
    double vhe3p, vhe3n;

    // Cosmic ray (48 bytes) - matches PackedCosmicRay order
    double cr1, cr2, cr3, cr4, cr5, cr6;

    // Ice gradients (72 bytes + 8 pad)
    double icegrad0, icegrad1, icegrad2, icegrad3, icegrad4;
    double icegrad5, icegrad6, icegrad7, icegrad8;
    double _pad1;

    // Detector systematics (16 bytes)
    double deltaDomeff;
    double holeiceFwd;

    // Astrophysical (32 bytes)
    double astroNorm;
    double astroDgamma;
    double astroDgammaSec;
    double astroPivot;

    // Cross sections (16 bytes)
    double neuaneuRatio;
    double nuxs;
    double nubarxs;
    double _pad2;
};

//==============================================================================
// Device Functions for Packed Data Access
//==============================================================================

/**
 * @brief Load hadronic data using vectorized loads
 */
__device__ __forceinline__ void loadHadronicVectorized(
    const PackedHadronic* __restrict__ src,
    double& HEkp, double& HEkm,
    double& VHE1pip, double& VHE1pim,
    double& VHE3kp, double& VHE3km,
    double& VHE3pip, double& VHE3pim,
    double& VHE3p, double& VHE3n
) {
    // Load as 5x double2 for coalesced access
    const double2* src2 = reinterpret_cast<const double2*>(src);

    double2 v0 = __ldg(&src2[0]);
    double2 v1 = __ldg(&src2[1]);
    double2 v2 = __ldg(&src2[2]);
    double2 v3 = __ldg(&src2[3]);
    double2 v4 = __ldg(&src2[4]);

    HEkp = v0.x;    HEkm = v0.y;
    VHE1pip = v1.x; VHE1pim = v1.y;
    VHE3kp = v2.x;  VHE3km = v2.y;
    VHE3pip = v3.x; VHE3pim = v3.y;
    VHE3p = v4.x;   VHE3n = v4.y;
}

/**
 * @brief Load cosmic ray data using vectorized loads
 */
__device__ __forceinline__ void loadCosmicRayVectorized(
    const PackedCosmicRay* __restrict__ src,
    double& cr1, double& cr2, double& cr3,
    double& cr4, double& cr5, double& cr6
) {
    const double2* src2 = reinterpret_cast<const double2*>(src);

    double2 v0 = __ldg(&src2[0]);
    double2 v1 = __ldg(&src2[1]);
    double2 v2 = __ldg(&src2[2]);

    cr1 = v0.x; cr2 = v0.y;
    cr3 = v1.x; cr4 = v1.y;
    cr5 = v2.x; cr6 = v2.y;
}

/**
 * @brief Load atmospheric data using vectorized load
 */
__device__ __forceinline__ void loadAtmosphericVectorized(
    const PackedAtmospheric* __restrict__ src,
    double& atmDensity, double& kaonLosses
) {
    double2 v = __ldg(reinterpret_cast<const double2*>(src));
    atmDensity = v.x;
    kaonLosses = v.y;
}

/**
 * @brief Load ice gradient data using vectorized loads
 */
__device__ __forceinline__ void loadIceGradVectorized(
    const PackedIceGrad* __restrict__ src,
    double grads[9]
) {
    const double2* src2 = reinterpret_cast<const double2*>(src);

    double2 v0 = __ldg(&src2[0]);
    double2 v1 = __ldg(&src2[1]);
    double2 v2 = __ldg(&src2[2]);
    double2 v3 = __ldg(&src2[3]);
    double2 v4 = __ldg(&src2[4]);

    grads[0] = v0.x; grads[1] = v0.y;
    grads[2] = v1.x; grads[3] = v1.y;
    grads[4] = v2.x; grads[5] = v2.y;
    grads[6] = v3.x; grads[7] = v3.y;
    grads[8] = v4.x;  // v4.y is padding
}

/**
 * @brief Compute hadronic correction using FMA operations
 *
 * Uses fused multiply-add for better numerical accuracy and performance.
 */
__device__ __forceinline__ double computeHadronicFMA(
    double HEkp, double HEkm,
    double VHE1pip, double VHE1pim,
    double VHE3kp, double VHE3km,
    double VHE3pip, double VHE3pim,
    double VHE3p, double VHE3n,
    double pHEkp, double pHEkm,
    double pVHE1pip, double pVHE1pim,
    double pVHE3kp, double pVHE3km,
    double pVHE3pip, double pVHE3pim,
    double pVHE3p, double pVHE3n
) {
    double result = 0.0;
    result = fma(pHEkp, HEkp, result);
    result = fma(pHEkm, HEkm, result);
    result = fma(pVHE1pip, VHE1pip, result);
    result = fma(pVHE1pim, VHE1pim, result);
    result = fma(pVHE3kp, VHE3kp, result);
    result = fma(pVHE3km, VHE3km, result);
    result = fma(pVHE3pip, VHE3pip, result);
    result = fma(pVHE3pim, VHE3pim, result);
    result = fma(pVHE3p, VHE3p, result);
    result = fma(pVHE3n, VHE3n, result);
    return result;
}

/**
 * @brief Compute cosmic ray correction using FMA operations
 */
__device__ __forceinline__ double computeCosmicRayFMA(
    double cr1, double cr2, double cr3,
    double cr4, double cr5, double cr6,
    double pCR1, double pCR2, double pCR3,
    double pCR4, double pCR5, double pCR6
) {
    double result = 0.0;
    result = fma(pCR1, cr1, result);
    result = fma(pCR2, cr2, result);
    result = fma(pCR3, cr3, result);
    result = fma(pCR4, cr4, result);
    result = fma(pCR5, cr5, result);
    result = fma(pCR6, cr6, result);
    return result;
}

/**
 * @brief Compute ice gradient weight using FMA operations
 */
__device__ __forceinline__ double computeIceGradFMA(
    const double grads[9],
    double p0, double p1, double p2,
    double p3, double p4, double p5,
    double p6, double p7, double p8
) {
    double weight = 1.0;
    weight *= fma(p0, grads[0], 1.0);
    weight *= fma(p1, grads[1], 1.0);
    weight *= fma(p2, grads[2], 1.0);
    weight *= fma(p3, grads[3], 1.0);
    weight *= fma(p4, grads[4], 1.0);
    weight *= fma(p5, grads[5], 1.0);
    weight *= fma(p6, grads[6], 1.0);
    weight *= fma(p7, grads[7], 1.0);
    weight *= fma(p8, grads[8], 1.0);
    return weight;
}

//==============================================================================
// GPU Packed Data Manager
//==============================================================================

/**
 * @brief Manager for packed event data on GPU
 *
 * Handles conversion from SoA to packed format and memory management.
 */
class GPUPackedDataManager {
public:
    GPUPackedDataManager() : d_packedEvents_(nullptr), d_packedParams_(nullptr),
                             numEvents_(0), initialized_(false) {}

    ~GPUPackedDataManager() {
        deallocate();
    }

    // Non-copyable
    GPUPackedDataManager(const GPUPackedDataManager&) = delete;
    GPUPackedDataManager& operator=(const GPUPackedDataManager&) = delete;

    /**
     * @brief Allocate packed data from existing SoA data
     */
    void createFromSoA(const GPUEventDataSoA& soaData, cudaStream_t stream = nullptr);

    /**
     * @brief Update packed parameters from raw parameter array
     */
    void updateParams(const double* h_params, cudaStream_t stream = nullptr);

    /**
     * @brief Get device pointer to packed events
     */
    PackedEventData* getPackedEvents() { return d_packedEvents_; }
    const PackedEventData* getPackedEvents() const { return d_packedEvents_; }

    /**
     * @brief Get device pointer to packed parameters
     */
    PackedParams* getPackedParams() { return d_packedParams_; }
    const PackedParams* getPackedParams() const { return d_packedParams_; }

    /**
     * @brief Get number of events
     */
    size_t getNumEvents() const { return numEvents_; }

    /**
     * @brief Get memory usage
     */
    size_t getMemoryUsage() const {
        return numEvents_ * sizeof(PackedEventData) + sizeof(PackedParams);
    }

    void deallocate() {
        if (d_packedEvents_) {
            cudaFree(d_packedEvents_);
            d_packedEvents_ = nullptr;
        }
        if (d_packedParams_) {
            cudaFree(d_packedParams_);
            d_packedParams_ = nullptr;
        }
        initialized_ = false;
    }

private:
    PackedEventData* d_packedEvents_;
    PackedParams* d_packedParams_;
    size_t numEvents_;
    bool initialized_;
};

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA

#endif // GOLLUMFIT_GPU_EVENT_DATA_PACKED_H
