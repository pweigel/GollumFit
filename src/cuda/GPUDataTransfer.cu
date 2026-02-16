/**
 * @file GPUDataTransfer.cu
 * @brief Implementation of GPU event data transfer and management.
 *
 * This file implements the GPUEventDataManager class which handles:
 * - Memory allocation on GPU
 * - AoS to SoA data transformation
 * - CPU to GPU data transfers
 * - Bin index computation
 */

#include "cuda/GPUEventData.h"
#include <algorithm>
#include <cmath>

#ifdef GOLLUMFIT_USE_CUDA

namespace gollumfit {
namespace gpu {

//==============================================================================
// GPUEventDataManager Implementation
//==============================================================================

GPUEventDataManager::GPUEventDataManager()
    : deviceData_{}, numEvents_(0), initialized_(false) {
    // Zero out all pointers in deviceData_
    std::memset(&deviceData_, 0, sizeof(GPUEventDataSoA));
}

GPUEventDataManager::~GPUEventDataManager() {
    deallocate();
}

GPUEventDataManager::GPUEventDataManager(GPUEventDataManager&& other) noexcept
    : deviceData_(other.deviceData_),
      numEvents_(other.numEvents_),
      initialized_(other.initialized_) {
    // Clear other's data to prevent double-free
    std::memset(&other.deviceData_, 0, sizeof(GPUEventDataSoA));
    other.numEvents_ = 0;
    other.initialized_ = false;
}

GPUEventDataManager& GPUEventDataManager::operator=(GPUEventDataManager&& other) noexcept {
    if (this != &other) {
        deallocate();
        deviceData_ = other.deviceData_;
        numEvents_ = other.numEvents_;
        initialized_ = other.initialized_;

        std::memset(&other.deviceData_, 0, sizeof(GPUEventDataSoA));
        other.numEvents_ = 0;
        other.initialized_ = false;
    }
    return *this;
}

void GPUEventDataManager::allocate(size_t numEvents) {
    if (numEvents == numEvents_ && initialized_) {
        return;  // Already allocated with same size
    }

    // Free existing allocations
    deallocate();

    numEvents_ = numEvents;

    if (numEvents == 0) {
        return;
    }

    // Allocate primary physics arrays (float)
    allocateArray(deviceData_.energy, numEvents);
    allocateArray(deviceData_.zenith, numEvents);
    allocateArray(deviceData_.primaryEnergy, numEvents);
    allocateArray(deviceData_.primaryZenith, numEvents);
    allocateArray(deviceData_.primaryAzimuth, numEvents);
    allocateArray(deviceData_.totalColumnDepth, numEvents);
    allocateArray(deviceData_.intX, numEvents);
    allocateArray(deviceData_.intY, numEvents);

    // Allocate discrete arrays
    allocateArray(deviceData_.topology, numEvents);
    allocateArray(deviceData_.primaryType, numEvents);
    allocateArray(deviceData_.numEvents, numEvents);

    // Allocate flux weight arrays (double)
    allocateArray(deviceData_.cachedConvWeight, numEvents);
    allocateArray(deviceData_.cachedPromptWeight, numEvents);
    allocateArray(deviceData_.cachedAstroWeight, numEvents);
    allocateArray(deviceData_.cachedWeight, numEvents);

    // Allocate hole ice arrays (double)
    allocateArray(deviceData_.cachedHoleIceConv, numEvents);
    allocateArray(deviceData_.cachedHoleIcePrompt, numEvents);
    allocateArray(deviceData_.cachedHoleIceAstro, numEvents);

    // Allocate DOM efficiency arrays (double)
    allocateArray(deviceData_.cachedDOMEffConv, numEvents);
    allocateArray(deviceData_.cachedDOMEffPrompt, numEvents);
    allocateArray(deviceData_.cachedDOMEffAstro, numEvents);

    // Allocate hadronic arrays (double)
    allocateArray(deviceData_.cachedHadronicHEkp, numEvents);
    allocateArray(deviceData_.cachedHadronicHEkm, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE1pip, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE1pim, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE3kp, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE3km, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE3pip, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE3pim, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE3p, numEvents);
    allocateArray(deviceData_.cachedHadronicVHE3n, numEvents);

    // Allocate cosmic ray arrays (double)
    allocateArray(deviceData_.cachedCosmicRay1, numEvents);
    allocateArray(deviceData_.cachedCosmicRay2, numEvents);
    allocateArray(deviceData_.cachedCosmicRay3, numEvents);
    allocateArray(deviceData_.cachedCosmicRay4, numEvents);
    allocateArray(deviceData_.cachedCosmicRay5, numEvents);
    allocateArray(deviceData_.cachedCosmicRay6, numEvents);

    // Allocate atmospheric arrays (double)
    allocateArray(deviceData_.cachedAtmDensity, numEvents);
    allocateArray(deviceData_.cachedKaonLosses, numEvents);

    // Allocate ice gradient arrays (double)
    allocateArray(deviceData_.cachedIceGrad0, numEvents);
    allocateArray(deviceData_.cachedIceGrad1, numEvents);
    allocateArray(deviceData_.cachedIceGrad2, numEvents);
    allocateArray(deviceData_.cachedIceGrad3, numEvents);
    allocateArray(deviceData_.cachedIceGrad4, numEvents);
    allocateArray(deviceData_.cachedIceGrad5, numEvents);
    allocateArray(deviceData_.cachedIceGrad6, numEvents);
    allocateArray(deviceData_.cachedIceGrad7, numEvents);
    allocateArray(deviceData_.cachedIceGrad8, numEvents);

    // Allocate precomputed transcendentals (double)
    allocateArray(deviceData_.log10Energy, numEvents);
    allocateArray(deviceData_.cosZenith, numEvents);
    allocateArray(deviceData_.log10PrimaryEnergy, numEvents);
    allocateArray(deviceData_.cosPrimaryZenith, numEvents);

    // Allocate cached spline basis — DOM efficiency
    allocateArray(deviceData_.cachedDOMEffSpan0, numEvents);
    allocateArray(deviceData_.cachedDOMEffSpan1, numEvents);
    allocateArray(deviceData_.cachedDOMEffBasis00, numEvents);
    allocateArray(deviceData_.cachedDOMEffBasis01, numEvents);
    allocateArray(deviceData_.cachedDOMEffBasis02, numEvents);
    allocateArray(deviceData_.cachedDOMEffBasis10, numEvents);
    allocateArray(deviceData_.cachedDOMEffBasis11, numEvents);
    allocateArray(deviceData_.cachedDOMEffBasis12, numEvents);

    // Allocate cached spline basis — Hole ice
    allocateArray(deviceData_.cachedHoleIceSpan0, numEvents);
    allocateArray(deviceData_.cachedHoleIceSpan1, numEvents);
    allocateArray(deviceData_.cachedHoleIceBasis00, numEvents);
    allocateArray(deviceData_.cachedHoleIceBasis01, numEvents);
    allocateArray(deviceData_.cachedHoleIceBasis02, numEvents);
    allocateArray(deviceData_.cachedHoleIceBasis10, numEvents);
    allocateArray(deviceData_.cachedHoleIceBasis11, numEvents);
    allocateArray(deviceData_.cachedHoleIceBasis12, numEvents);

    // Allocate bin index array
    allocateArray(deviceData_.binIndex, numEvents);

    deviceData_.numEvents_total = numEvents;
}

void GPUEventDataManager::deallocate() {
    if (numEvents_ == 0) {
        return;
    }

    // Free primary physics arrays
    freeArray(deviceData_.energy);
    freeArray(deviceData_.zenith);
    freeArray(deviceData_.primaryEnergy);
    freeArray(deviceData_.primaryZenith);
    freeArray(deviceData_.primaryAzimuth);
    freeArray(deviceData_.totalColumnDepth);
    freeArray(deviceData_.intX);
    freeArray(deviceData_.intY);

    // Free discrete arrays
    freeArray(deviceData_.topology);
    freeArray(deviceData_.primaryType);
    freeArray(deviceData_.numEvents);

    // Free flux weight arrays
    freeArray(deviceData_.cachedConvWeight);
    freeArray(deviceData_.cachedPromptWeight);
    freeArray(deviceData_.cachedAstroWeight);
    freeArray(deviceData_.cachedWeight);

    // Free hole ice arrays
    freeArray(deviceData_.cachedHoleIceConv);
    freeArray(deviceData_.cachedHoleIcePrompt);
    freeArray(deviceData_.cachedHoleIceAstro);

    // Free DOM efficiency arrays
    freeArray(deviceData_.cachedDOMEffConv);
    freeArray(deviceData_.cachedDOMEffPrompt);
    freeArray(deviceData_.cachedDOMEffAstro);

    // Free hadronic arrays
    freeArray(deviceData_.cachedHadronicHEkp);
    freeArray(deviceData_.cachedHadronicHEkm);
    freeArray(deviceData_.cachedHadronicVHE1pip);
    freeArray(deviceData_.cachedHadronicVHE1pim);
    freeArray(deviceData_.cachedHadronicVHE3kp);
    freeArray(deviceData_.cachedHadronicVHE3km);
    freeArray(deviceData_.cachedHadronicVHE3pip);
    freeArray(deviceData_.cachedHadronicVHE3pim);
    freeArray(deviceData_.cachedHadronicVHE3p);
    freeArray(deviceData_.cachedHadronicVHE3n);

    // Free cosmic ray arrays
    freeArray(deviceData_.cachedCosmicRay1);
    freeArray(deviceData_.cachedCosmicRay2);
    freeArray(deviceData_.cachedCosmicRay3);
    freeArray(deviceData_.cachedCosmicRay4);
    freeArray(deviceData_.cachedCosmicRay5);
    freeArray(deviceData_.cachedCosmicRay6);

    // Free atmospheric arrays
    freeArray(deviceData_.cachedAtmDensity);
    freeArray(deviceData_.cachedKaonLosses);

    // Free ice gradient arrays
    freeArray(deviceData_.cachedIceGrad0);
    freeArray(deviceData_.cachedIceGrad1);
    freeArray(deviceData_.cachedIceGrad2);
    freeArray(deviceData_.cachedIceGrad3);
    freeArray(deviceData_.cachedIceGrad4);
    freeArray(deviceData_.cachedIceGrad5);
    freeArray(deviceData_.cachedIceGrad6);
    freeArray(deviceData_.cachedIceGrad7);
    freeArray(deviceData_.cachedIceGrad8);

    // Free precomputed transcendentals
    freeArray(deviceData_.log10Energy);
    freeArray(deviceData_.cosZenith);
    freeArray(deviceData_.log10PrimaryEnergy);
    freeArray(deviceData_.cosPrimaryZenith);

    // Free cached spline basis — DOM efficiency
    freeArray(deviceData_.cachedDOMEffSpan0);
    freeArray(deviceData_.cachedDOMEffSpan1);
    freeArray(deviceData_.cachedDOMEffBasis00);
    freeArray(deviceData_.cachedDOMEffBasis01);
    freeArray(deviceData_.cachedDOMEffBasis02);
    freeArray(deviceData_.cachedDOMEffBasis10);
    freeArray(deviceData_.cachedDOMEffBasis11);
    freeArray(deviceData_.cachedDOMEffBasis12);

    // Free cached spline basis — Hole ice
    freeArray(deviceData_.cachedHoleIceSpan0);
    freeArray(deviceData_.cachedHoleIceSpan1);
    freeArray(deviceData_.cachedHoleIceBasis00);
    freeArray(deviceData_.cachedHoleIceBasis01);
    freeArray(deviceData_.cachedHoleIceBasis02);
    freeArray(deviceData_.cachedHoleIceBasis10);
    freeArray(deviceData_.cachedHoleIceBasis11);
    freeArray(deviceData_.cachedHoleIceBasis12);

    // Free bin index array
    freeArray(deviceData_.binIndex);

    deviceData_.numEvents_total = 0;
    numEvents_ = 0;
    initialized_ = false;
}

void GPUEventDataManager::uploadFromCPU(const std::deque<Event>& events,
                                         cudaStream_t stream) {
    uploadFromCPUImpl(events, stream);
}

void GPUEventDataManager::uploadFromCPU(const std::vector<Event>& events,
                                         cudaStream_t stream) {
    uploadFromCPUImpl(events, stream);
}

template<typename Container>
void GPUEventDataManager::uploadFromCPUImpl(const Container& events,
                                             cudaStream_t stream) {
    const size_t numEvents = events.size();
    if (numEvents == 0) {
        return;
    }

    // Allocate if needed
    if (numEvents_ != numEvents) {
        allocate(numEvents);
    }

    // Create temporary host arrays for batch copy
    // Using pinned memory for faster transfer
    std::vector<float> h_energy(numEvents);
    std::vector<float> h_zenith(numEvents);
    std::vector<float> h_primaryEnergy(numEvents);
    std::vector<float> h_primaryZenith(numEvents);
    std::vector<float> h_primaryAzimuth(numEvents);
    std::vector<float> h_totalColumnDepth(numEvents);
    std::vector<float> h_intX(numEvents);
    std::vector<float> h_intY(numEvents);
    std::vector<uint32_t> h_topology(numEvents);
    std::vector<int32_t> h_primaryType(numEvents);
    std::vector<int32_t> h_numEvents(numEvents);

    std::vector<double> h_cachedConvWeight(numEvents);
    std::vector<double> h_cachedPromptWeight(numEvents);
    std::vector<double> h_cachedAstroWeight(numEvents);
    std::vector<double> h_cachedWeight(numEvents);

    std::vector<double> h_cachedHoleIceConv(numEvents);
    std::vector<double> h_cachedHoleIcePrompt(numEvents);
    std::vector<double> h_cachedHoleIceAstro(numEvents);
    std::vector<double> h_cachedDOMEffConv(numEvents);
    std::vector<double> h_cachedDOMEffPrompt(numEvents);
    std::vector<double> h_cachedDOMEffAstro(numEvents);

    // MIXED PRECISION: Hadronic, cosmic ray, atmospheric, ice gradients use float
    std::vector<float> h_cachedHadronicHEkp(numEvents);
    std::vector<float> h_cachedHadronicHEkm(numEvents);
    std::vector<float> h_cachedHadronicVHE1pip(numEvents);
    std::vector<float> h_cachedHadronicVHE1pim(numEvents);
    std::vector<float> h_cachedHadronicVHE3kp(numEvents);
    std::vector<float> h_cachedHadronicVHE3km(numEvents);
    std::vector<float> h_cachedHadronicVHE3pip(numEvents);
    std::vector<float> h_cachedHadronicVHE3pim(numEvents);
    std::vector<float> h_cachedHadronicVHE3p(numEvents);
    std::vector<float> h_cachedHadronicVHE3n(numEvents);

    std::vector<float> h_cachedCosmicRay1(numEvents);
    std::vector<float> h_cachedCosmicRay2(numEvents);
    std::vector<float> h_cachedCosmicRay3(numEvents);
    std::vector<float> h_cachedCosmicRay4(numEvents);
    std::vector<float> h_cachedCosmicRay5(numEvents);
    std::vector<float> h_cachedCosmicRay6(numEvents);

    std::vector<float> h_cachedAtmDensity(numEvents);
    std::vector<float> h_cachedKaonLosses(numEvents);

    std::vector<float> h_cachedIceGrad0(numEvents);
    std::vector<float> h_cachedIceGrad1(numEvents);
    std::vector<float> h_cachedIceGrad2(numEvents);
    std::vector<float> h_cachedIceGrad3(numEvents);
    std::vector<float> h_cachedIceGrad4(numEvents);
    std::vector<float> h_cachedIceGrad5(numEvents);
    std::vector<float> h_cachedIceGrad6(numEvents);
    std::vector<float> h_cachedIceGrad7(numEvents);
    std::vector<float> h_cachedIceGrad8(numEvents);

    // Transform AoS to SoA
    size_t i = 0;
    for (const auto& e : events) {
        h_energy[i] = e.energy;
        h_zenith[i] = e.zenith;
        h_primaryEnergy[i] = e.primaryEnergy;
        h_primaryZenith[i] = e.primaryZenith;
        h_primaryAzimuth[i] = e.primaryAzimuth;
        h_totalColumnDepth[i] = e.totalColumnDepth;
        h_intX[i] = e.intX;
        h_intY[i] = e.intY;
        h_topology[i] = e.topology;
        h_primaryType[i] = static_cast<int32_t>(e.primaryType);
        h_numEvents[i] = e.num_events;

        h_cachedConvWeight[i] = e.cachedConvWeight;
        h_cachedPromptWeight[i] = e.cachedPromptWeight;
        h_cachedAstroWeight[i] = e.cachedAstroWeight;
        h_cachedWeight[i] = e.cachedWeight;

        h_cachedHoleIceConv[i] = e.cachedHoleIceConv;
        h_cachedHoleIcePrompt[i] = e.cachedHoleIcePrompt;
        h_cachedHoleIceAstro[i] = e.cachedHoleIceAstro;
        h_cachedDOMEffConv[i] = e.cachedDOMEffConv;
        h_cachedDOMEffPrompt[i] = e.cachedDOMEffPrompt;
        h_cachedDOMEffAstro[i] = e.cachedDOMEffAstro;

        // MIXED PRECISION: Convert from double (Event) to float (GPU)
        h_cachedHadronicHEkp[i] = static_cast<float>(e.cachedHadronicHEkp);
        h_cachedHadronicHEkm[i] = static_cast<float>(e.cachedHadronicHEkm);
        h_cachedHadronicVHE1pip[i] = static_cast<float>(e.cachedHadronicVHE1pip);
        h_cachedHadronicVHE1pim[i] = static_cast<float>(e.cachedHadronicVHE1pim);
        h_cachedHadronicVHE3kp[i] = static_cast<float>(e.cachedHadronicVHE3kp);
        h_cachedHadronicVHE3km[i] = static_cast<float>(e.cachedHadronicVHE3km);
        h_cachedHadronicVHE3pip[i] = static_cast<float>(e.cachedHadronicVHE3pip);
        h_cachedHadronicVHE3pim[i] = static_cast<float>(e.cachedHadronicVHE3pim);
        h_cachedHadronicVHE3p[i] = static_cast<float>(e.cachedHadronicVHE3p);
        h_cachedHadronicVHE3n[i] = static_cast<float>(e.cachedHadronicVHE3n);

        h_cachedCosmicRay1[i] = static_cast<float>(e.cachedCosmicRay1);
        h_cachedCosmicRay2[i] = static_cast<float>(e.cachedCosmicRay2);
        h_cachedCosmicRay3[i] = static_cast<float>(e.cachedCosmicRay3);
        h_cachedCosmicRay4[i] = static_cast<float>(e.cachedCosmicRay4);
        h_cachedCosmicRay5[i] = static_cast<float>(e.cachedCosmicRay5);
        h_cachedCosmicRay6[i] = static_cast<float>(e.cachedCosmicRay6);

        h_cachedAtmDensity[i] = static_cast<float>(e.cachedAtmDensity);
        h_cachedKaonLosses[i] = static_cast<float>(e.cachedKaonLosses);

        h_cachedIceGrad0[i] = static_cast<float>(e.cachedIceGrad0);
        h_cachedIceGrad1[i] = static_cast<float>(e.cachedIceGrad1);
        h_cachedIceGrad2[i] = static_cast<float>(e.cachedIceGrad2);
        h_cachedIceGrad3[i] = static_cast<float>(e.cachedIceGrad3);
        h_cachedIceGrad4[i] = static_cast<float>(e.cachedIceGrad4);
        h_cachedIceGrad5[i] = static_cast<float>(e.cachedIceGrad5);
        h_cachedIceGrad6[i] = static_cast<float>(e.cachedIceGrad6);
        h_cachedIceGrad7[i] = static_cast<float>(e.cachedIceGrad7);
        h_cachedIceGrad8[i] = static_cast<float>(e.cachedIceGrad8);

        ++i;
    }

    // Transfer to GPU
    auto copyToDevice = [stream](void* dst, const void* src, size_t bytes) {
        if (stream) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        }
    };

    // Float arrays
    copyToDevice(deviceData_.energy, h_energy.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.zenith, h_zenith.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.primaryEnergy, h_primaryEnergy.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.primaryZenith, h_primaryZenith.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.primaryAzimuth, h_primaryAzimuth.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.totalColumnDepth, h_totalColumnDepth.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.intX, h_intX.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.intY, h_intY.data(), numEvents * sizeof(float));

    // Integer arrays
    copyToDevice(deviceData_.topology, h_topology.data(), numEvents * sizeof(uint32_t));
    copyToDevice(deviceData_.primaryType, h_primaryType.data(), numEvents * sizeof(int32_t));
    copyToDevice(deviceData_.numEvents, h_numEvents.data(), numEvents * sizeof(int32_t));

    // Double arrays - flux weights
    copyToDevice(deviceData_.cachedConvWeight, h_cachedConvWeight.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedPromptWeight, h_cachedPromptWeight.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedAstroWeight, h_cachedAstroWeight.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedWeight, h_cachedWeight.data(), numEvents * sizeof(double));

    // Double arrays - detector systematics
    copyToDevice(deviceData_.cachedHoleIceConv, h_cachedHoleIceConv.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedHoleIcePrompt, h_cachedHoleIcePrompt.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedHoleIceAstro, h_cachedHoleIceAstro.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedDOMEffConv, h_cachedDOMEffConv.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedDOMEffPrompt, h_cachedDOMEffPrompt.data(), numEvents * sizeof(double));
    copyToDevice(deviceData_.cachedDOMEffAstro, h_cachedDOMEffAstro.data(), numEvents * sizeof(double));

    // Float arrays - hadronic (MIXED PRECISION)
    copyToDevice(deviceData_.cachedHadronicHEkp, h_cachedHadronicHEkp.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicHEkm, h_cachedHadronicHEkm.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE1pip, h_cachedHadronicVHE1pip.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE1pim, h_cachedHadronicVHE1pim.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE3kp, h_cachedHadronicVHE3kp.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE3km, h_cachedHadronicVHE3km.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE3pip, h_cachedHadronicVHE3pip.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE3pim, h_cachedHadronicVHE3pim.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE3p, h_cachedHadronicVHE3p.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedHadronicVHE3n, h_cachedHadronicVHE3n.data(), numEvents * sizeof(float));

    // Float arrays - cosmic ray (MIXED PRECISION)
    copyToDevice(deviceData_.cachedCosmicRay1, h_cachedCosmicRay1.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedCosmicRay2, h_cachedCosmicRay2.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedCosmicRay3, h_cachedCosmicRay3.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedCosmicRay4, h_cachedCosmicRay4.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedCosmicRay5, h_cachedCosmicRay5.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedCosmicRay6, h_cachedCosmicRay6.data(), numEvents * sizeof(float));

    // Float arrays - atmospheric (MIXED PRECISION)
    copyToDevice(deviceData_.cachedAtmDensity, h_cachedAtmDensity.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedKaonLosses, h_cachedKaonLosses.data(), numEvents * sizeof(float));

    // Float arrays - ice gradients (MIXED PRECISION)
    copyToDevice(deviceData_.cachedIceGrad0, h_cachedIceGrad0.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad1, h_cachedIceGrad1.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad2, h_cachedIceGrad2.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad3, h_cachedIceGrad3.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad4, h_cachedIceGrad4.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad5, h_cachedIceGrad5.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad6, h_cachedIceGrad6.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad7, h_cachedIceGrad7.data(), numEvents * sizeof(float));
    copyToDevice(deviceData_.cachedIceGrad8, h_cachedIceGrad8.data(), numEvents * sizeof(float));

    // Initialize bin indices to -1 (will be computed separately)
    CUDA_CHECK(cudaMemset(deviceData_.binIndex, -1, numEvents * sizeof(int32_t)));

    initialized_ = true;
}

void GPUEventDataManager::computeBinIndices(const std::vector<double>& energyEdges,
                                             const std::vector<double>& zenithEdges,
                                             int numTopologies) {
    if (!initialized_ || numEvents_ == 0) {
        return;
    }

    // This would typically be done with a CUDA kernel for efficiency
    // For now, we'll do it on CPU and upload (can be optimized later)

    const int numEnergyBins = static_cast<int>(energyEdges.size()) - 1;
    const int numZenithBins = static_cast<int>(zenithEdges.size()) - 1;

    // Download event data needed for binning
    std::vector<float> h_energy(numEvents_);
    std::vector<float> h_zenith(numEvents_);
    std::vector<uint32_t> h_topology(numEvents_);
    std::vector<int32_t> h_binIndex(numEvents_);

    CUDA_CHECK(cudaMemcpy(h_energy.data(), deviceData_.energy,
                          numEvents_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_zenith.data(), deviceData_.zenith,
                          numEvents_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topology.data(), deviceData_.topology,
                          numEvents_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Compute bin indices
    for (size_t i = 0; i < numEvents_; ++i) {
        float e = h_energy[i];
        float cz = std::cos(h_zenith[i]);
        int topo = static_cast<int>(h_topology[i]);

        // Find energy bin (log scale assumed)
        int eBin = -1;
        for (int j = 0; j < numEnergyBins; ++j) {
            if (e >= energyEdges[j] && e < energyEdges[j + 1]) {
                eBin = j;
                break;
            }
        }

        // Find zenith bin (cos(zenith))
        int zBin = -1;
        for (int j = 0; j < numZenithBins; ++j) {
            if (cz >= zenithEdges[j] && cz < zenithEdges[j + 1]) {
                zBin = j;
                break;
            }
        }

        // Compute flattened index
        if (eBin >= 0 && zBin >= 0 && topo >= 0 && topo < numTopologies) {
            // 3D index: [energy][zenith][topology]
            h_binIndex[i] = eBin * numZenithBins * numTopologies +
                            zBin * numTopologies + topo;
        } else {
            h_binIndex[i] = -1;  // Out of bounds
        }
    }

    // Upload bin indices
    CUDA_CHECK(cudaMemcpy(deviceData_.binIndex, h_binIndex.data(),
                          numEvents_ * sizeof(int32_t), cudaMemcpyHostToDevice));
}

size_t GPUEventDataManager::getMemoryUsage() const {
    if (numEvents_ == 0) {
        return 0;
    }

    size_t usage = 0;

    // Float arrays - primary physics (8 arrays)
    usage += 8 * numEvents_ * sizeof(float);

    // Integer arrays (3 arrays)
    usage += numEvents_ * sizeof(uint32_t);  // topology
    usage += 2 * numEvents_ * sizeof(int32_t);  // primaryType, numEvents

    // Double arrays - flux weights and detector systematics (10 arrays)
    // Flux weights: 4
    // Hole ice: 3
    // DOM eff: 3
    usage += 10 * numEvents_ * sizeof(double);

    // Float arrays (MIXED PRECISION optimization) - 27 arrays total
    // Hadronic: 10
    // Cosmic ray: 6
    // Atmospheric: 2
    // Ice gradients: 9
    usage += 27 * numEvents_ * sizeof(float);

    // Precomputed transcendentals (4 double arrays)
    usage += 4 * numEvents_ * sizeof(double);

    // Cached spline basis: 2 types × (2 int32 spans + 6 float basis) = 4 int32 + 12 float
    usage += 4 * numEvents_ * sizeof(int32_t);
    usage += 12 * numEvents_ * sizeof(float);

    // Bin index
    usage += numEvents_ * sizeof(int32_t);

    return usage;
}

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA
