/**
 * @file GPUEventData.h
 * @brief Structure-of-Arrays (SoA) event data structure for GPU acceleration.
 *
 * This file defines the GPU-optimized event data layout that transforms the
 * Array-of-Structures (AoS) Event class from Event.h into a Structure-of-Arrays
 * layout for optimal GPU memory coalescing.
 *
 * The SoA layout ensures that when threads in a warp access the same field
 * (e.g., all threads read energy), the memory accesses are coalesced into
 * a single efficient memory transaction.
 */

#ifndef GOLLUMFIT_GPU_EVENT_DATA_H
#define GOLLUMFIT_GPU_EVENT_DATA_H

#ifdef GOLLUMFIT_USE_CUDA

#include "GPUCommon.h"
#include "../Event.h"
#include <vector>
#include <deque>
#include <cstring>

namespace gollumfit {
namespace gpu {

//==============================================================================
// Forward Declarations
//==============================================================================

struct GPUEventDataSoA;
class GPUEventDataManager;

//==============================================================================
// GPU Event Data Structure (SoA)
//==============================================================================

/**
 * @brief Structure-of-Arrays layout for GPU event data.
 *
 * This structure mirrors the Event class fields but reorganized for optimal
 * GPU memory access patterns. Each array holds one field type for all events,
 * enabling coalesced memory access when GPU threads process events in parallel.
 *
 * Memory layout (for N events):
 * - Primary physics fields use float (4 bytes) for most reconstructed quantities
 * - Cached flux weights and detector corrections use double (8 bytes)
 * - Cached spline basis use float (4 bytes)
 * - Total memory per event: ~272 bytes
 */
struct GPUEventDataSoA {
    //--------------------------------------------------------------------------
    // Primary physics quantities (read-only after initialization)
    // Using float for reconstructed quantities (sufficient precision)
    //--------------------------------------------------------------------------

    float* energy;              ///< Reconstructed energy [N] (GeV, log scale binning)
    float* zenith;              ///< Reconstructed zenith angle [N] (radians)
    float* primaryEnergy;       ///< True primary neutrino energy [N] (GeV)
    float* primaryZenith;       ///< True primary zenith angle [N] (radians)
    float* primaryAzimuth;      ///< True primary azimuth angle [N] (radians)
    float* totalColumnDepth;    ///< Total column depth traversed [N] (g/cm^2)
    float* intX;                ///< Bjorken x [N]
    float* intY;                ///< Bjorken y [N]

    //--------------------------------------------------------------------------
    // Discrete quantities
    //--------------------------------------------------------------------------

    uint32_t* topology;         ///< Event topology [N] (0=cascade, 1=track, etc.)
    int32_t* primaryType;       ///< Primary particle type [N] (LW::ParticleType as int)
    int32_t* numEvents;         ///< Number of events combined (for meta-events) [N]

    //--------------------------------------------------------------------------
    // Cached flux weights (FP64 for numerical precision)
    // These are computed once at initialization from LeptonWeighter
    //--------------------------------------------------------------------------

    double* cachedConvWeight;   ///< Conventional atmospheric flux weight [N]
    double* cachedPromptWeight; ///< Prompt atmospheric flux weight [N]
    double* cachedAstroWeight;  ///< Astrophysical flux weight [N]
    double* cachedWeight;       ///< Data weight (usually 1.0 for real data) [N]

    //--------------------------------------------------------------------------
    // Cached detector systematic corrections
    // Stored as log10 of the correction factor
    //--------------------------------------------------------------------------

    // Hole ice corrections (3 flux components)
    double* cachedHoleIceConv;   ///< Hole ice correction for conventional flux [N]
    double* cachedHoleIcePrompt; ///< Hole ice correction for prompt flux [N]
    double* cachedHoleIceAstro;  ///< Hole ice correction for astro flux [N]

    // DOM efficiency corrections (3 flux components)
    double* cachedDOMEffConv;    ///< DOM efficiency correction for conventional flux [N]
    double* cachedDOMEffPrompt;  ///< DOM efficiency correction for prompt flux [N]
    double* cachedDOMEffAstro;   ///< DOM efficiency correction for astro flux [N]

    //--------------------------------------------------------------------------
    // Cached hadronic interaction parameters (10 DAEMONFLUX parameters)
    // MIXED PRECISION: Using float (FP32) for ~44% memory bandwidth reduction
    // These are relative corrections O(0.1), FP32's 7 significant digits suffice
    //--------------------------------------------------------------------------

    float* cachedHadronicHEkp;     ///< High energy K+ (158 GeV) [N]
    float* cachedHadronicHEkm;     ///< High energy K- (158 GeV) [N]
    float* cachedHadronicVHE1pip;  ///< Very high energy pi+ (20 TeV) [N]
    float* cachedHadronicVHE1pim;  ///< Very high energy pi- (20 TeV) [N]
    float* cachedHadronicVHE3kp;   ///< Very high energy K+ (2 PeV) [N]
    float* cachedHadronicVHE3km;   ///< Very high energy K- (2 PeV) [N]
    float* cachedHadronicVHE3pip;  ///< Very high energy pi+ (2 PeV) [N]
    float* cachedHadronicVHE3pim;  ///< Very high energy pi- (2 PeV) [N]
    float* cachedHadronicVHE3p;    ///< Very high energy p (2 PeV) [N]
    float* cachedHadronicVHE3n;    ///< Very high energy n (2 PeV) [N]

    //--------------------------------------------------------------------------
    // Cached cosmic ray parameters (6 GSF parameters)
    // MIXED PRECISION: Using float (FP32)
    //--------------------------------------------------------------------------

    float* cachedCosmicRay1;   ///< Cosmic ray parameter 1 [N]
    float* cachedCosmicRay2;   ///< Cosmic ray parameter 2 [N]
    float* cachedCosmicRay3;   ///< Cosmic ray parameter 3 [N]
    float* cachedCosmicRay4;   ///< Cosmic ray parameter 4 [N]
    float* cachedCosmicRay5;   ///< Cosmic ray parameter 5 [N]
    float* cachedCosmicRay6;   ///< Cosmic ray parameter 6 [N]

    //--------------------------------------------------------------------------
    // Cached atmospheric parameters
    // MIXED PRECISION: Using float (FP32)
    //--------------------------------------------------------------------------

    float* cachedAtmDensity;   ///< Atmospheric density uncertainty weight [N]
    float* cachedKaonLosses;   ///< Kaon energy losses uncertainty weight [N]

    //--------------------------------------------------------------------------
    // Cached ice gradient parameters (9 ice anisotropy parameters)
    // MIXED PRECISION: Using float (FP32)
    //--------------------------------------------------------------------------

    float* cachedIceGrad0;     ///< Ice gradient parameter 0 [N]
    float* cachedIceGrad1;     ///< Ice gradient parameter 1 [N]
    float* cachedIceGrad2;     ///< Ice gradient parameter 2 [N]
    float* cachedIceGrad3;     ///< Ice gradient parameter 3 [N]
    float* cachedIceGrad4;     ///< Ice gradient parameter 4 [N]
    float* cachedIceGrad5;     ///< Ice gradient parameter 5 [N]
    float* cachedIceGrad6;     ///< Ice gradient parameter 6 [N]
    float* cachedIceGrad7;     ///< Ice gradient parameter 7 [N]
    float* cachedIceGrad8;     ///< Ice gradient parameter 8 [N]

    //--------------------------------------------------------------------------
    // Precomputed transcendentals (computed once, used every kernel call)
    //--------------------------------------------------------------------------

    double* log10Energy;          ///< log10((double)energy) [N]
    double* cosZenith;            ///< cos((double)zenith) [N]
    double* log10PrimaryEnergy;   ///< log10((double)primaryEnergy) [N]
    double* cosPrimaryZenith;     ///< cos((double)primaryZenith) [N]

    //--------------------------------------------------------------------------
    // Cached spline basis for dims 0 & 1 — DOM efficiency
    // Assumes order <= 2 for dims 0 & 1 (3 non-zero basis functions)
    //--------------------------------------------------------------------------

    int32_t* cachedDOMEffSpan0;   ///< Knot span for dim 0; -1 if OOB [N]
    int32_t* cachedDOMEffSpan1;   ///< Knot span for dim 1; -1 if OOB [N]
    // MIXED PRECISION: Using float (FP32) — basis values in [0,1]
    float*   cachedDOMEffBasis00; ///< basis0[0] for all events [N]
    float*   cachedDOMEffBasis01; ///< basis0[1] [N]
    float*   cachedDOMEffBasis02; ///< basis0[2] [N]
    float*   cachedDOMEffBasis10; ///< basis1[0] [N]
    float*   cachedDOMEffBasis11; ///< basis1[1] [N]
    float*   cachedDOMEffBasis12; ///< basis1[2] [N]

    //--------------------------------------------------------------------------
    // Cached spline basis for dims 0 & 1 — Hole ice (same layout)
    //--------------------------------------------------------------------------

    int32_t* cachedHoleIceSpan0;   ///< Knot span for dim 0; -1 if OOB [N]
    int32_t* cachedHoleIceSpan1;   ///< Knot span for dim 1; -1 if OOB [N]
    // MIXED PRECISION: Using float (FP32) — basis values in [0,1]
    float*   cachedHoleIceBasis00; ///< basis0[0] for all events [N]
    float*   cachedHoleIceBasis01; ///< basis0[1] [N]
    float*   cachedHoleIceBasis02; ///< basis0[2] [N]
    float*   cachedHoleIceBasis10; ///< basis1[0] [N]
    float*   cachedHoleIceBasis11; ///< basis1[1] [N]
    float*   cachedHoleIceBasis12; ///< basis1[2] [N]

    //--------------------------------------------------------------------------
    // Precomputed bin indices for histogram accumulation
    //--------------------------------------------------------------------------

    int32_t* binIndex;          ///< Flattened 3D bin index [N] (-1 if out of bounds)

    //--------------------------------------------------------------------------
    // Metadata
    //--------------------------------------------------------------------------

    size_t numEvents_total;     ///< Total number of events stored

    /**
     * @brief Check if the structure is valid (has data)
     */
    __host__ __device__ bool isValid() const {
        return numEvents_total > 0 && energy != nullptr;
    }
};

//==============================================================================
// GPU Event Data Manager
//==============================================================================

/**
 * @brief Manager class for GPU event data lifecycle.
 *
 * This class handles:
 * - Allocation of GPU memory for all event arrays
 * - Transfer of data from CPU Event objects to GPU SoA format
 * - Transfer of results back to CPU
 * - Memory cleanup
 *
 * The manager maintains both device (GPU) and optionally pinned host memory
 * for efficient asynchronous transfers.
 */
class GPUEventDataManager {
public:
    /**
     * @brief Default constructor - creates empty manager
     */
    GPUEventDataManager();

    /**
     * @brief Destructor - frees all GPU memory
     */
    ~GPUEventDataManager();

    // Non-copyable
    GPUEventDataManager(const GPUEventDataManager&) = delete;
    GPUEventDataManager& operator=(const GPUEventDataManager&) = delete;

    // Movable
    GPUEventDataManager(GPUEventDataManager&& other) noexcept;
    GPUEventDataManager& operator=(GPUEventDataManager&& other) noexcept;

    /**
     * @brief Allocate GPU memory for the specified number of events
     * @param numEvents Number of events to allocate space for
     */
    void allocate(size_t numEvents);

    /**
     * @brief Free all GPU memory
     */
    void deallocate();

    /**
     * @brief Transfer events from CPU to GPU
     *
     * Converts AoS (Array-of-Structures) Event data to SoA (Structure-of-Arrays)
     * format and transfers to GPU memory.
     *
     * @param events Deque of CPU Event objects
     * @param stream CUDA stream for async transfer (nullptr for default stream)
     */
    void uploadFromCPU(const std::deque<Event>& events, cudaStream_t stream = nullptr);

    /**
     * @brief Transfer events from CPU vector to GPU
     * @param events Vector of CPU Event objects
     * @param stream CUDA stream for async transfer (nullptr for default stream)
     */
    void uploadFromCPU(const std::vector<Event>& events, cudaStream_t stream = nullptr);

    /**
     * @brief Set precomputed bin indices for all events
     *
     * @param energyEdges Energy bin edges (log10 scale)
     * @param zenithEdges Zenith bin edges (cos(zenith))
     * @param numTopologies Number of topology bins
     */
    void computeBinIndices(const std::vector<double>& energyEdges,
                           const std::vector<double>& zenithEdges,
                           int numTopologies);

    /**
     * @brief Get the GPU data structure (device pointers)
     */
    const GPUEventDataSoA& getDeviceData() const { return deviceData_; }
    GPUEventDataSoA& getDeviceData() { return deviceData_; }

    /**
     * @brief Get number of events
     */
    size_t getNumEvents() const { return numEvents_; }

    /**
     * @brief Check if data has been uploaded
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Get memory usage in bytes
     */
    size_t getMemoryUsage() const;

private:
    /**
     * @brief Internal helper to transfer data from a container
     */
    template<typename Container>
    void uploadFromCPUImpl(const Container& events, cudaStream_t stream);

    /**
     * @brief Allocate a single device array
     */
    template<typename T>
    void allocateArray(T*& ptr, size_t count);

    /**
     * @brief Free a single device array
     */
    template<typename T>
    void freeArray(T*& ptr);

    GPUEventDataSoA deviceData_;    ///< Structure holding device pointers
    size_t numEvents_;              ///< Number of events allocated
    bool initialized_;              ///< Whether data has been uploaded
};

//==============================================================================
// Inline Implementations
//==============================================================================

template<typename T>
void GPUEventDataManager::allocateArray(T*& ptr, size_t count) {
    if (count > 0) {
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    } else {
        ptr = nullptr;
    }
}

template<typename T>
void GPUEventDataManager::freeArray(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA

#endif // GOLLUMFIT_GPU_EVENT_DATA_H
