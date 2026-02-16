/**
 * @file precompute_reference_splines.cu
 * @brief CUDA kernel to precompute reference spline evaluations at upload time.
 *
 * The DOM efficiency and hole ice correction functions evaluate splines at fixed
 * reference values (DOM eff = 1.27, hole ice = -1.0) that only depend on per-event
 * log10(energy) and cos(zenith), which never change. By precomputing and caching
 * these 6 values per event once after spline upload, we eliminate 6 of 12 spline
 * evaluations per event in the weight and gradient kernels.
 *
 * Additionally precomputes:
 * - Transcendentals: log10(energy), cos(zenith), log10(primaryEnergy), cos(primaryZenith)
 * - Spline basis functions for dims 0 & 1 of DOM eff and hole ice splines,
 *   allowing the weight/gradient kernels to skip findKnotSpan + evaluateBasis
 *   for these two invariant dimensions and only evaluate dim 2 on the fly.
 */

#include "cuda/GPUEventData.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUSplineTable.h"
#include <cmath>

namespace gollumfit {
namespace gpu {

/**
 * @brief Kernel to precompute reference spline values, transcendentals,
 *        and cached basis functions for all events.
 *
 * One thread per event. For each event:
 * 1. Computes and stores transcendentals (log10Energy, cosZenith, etc.)
 * 2. Evaluates the 6 reference splines (3 DOM eff + 3 hole ice)
 * 3. If basisCacheValid: computes findKnotSpan + evaluateBasis for dims 0 & 1
 *    of the DOM eff and hole ice splines and caches the results
 */
__global__ void precomputeReferenceSplinesKernel(
    GPUEventDataSoA events,
    const GPUSplineLookup splines,
    const int numEvents
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Load per-event coordinates (same as weight kernel)
    const float energy = events.energy[tid];
    const float zenith = events.zenith[tid];
    const float primaryEnergy = events.primaryEnergy[tid];
    const float primaryZenith = events.primaryZenith[tid];
    const uint32_t topology = events.topology[tid];

    const double log10E = log10((double)energy);
    const double cosZ = cos((double)zenith);
    const double log10PE = log10((double)primaryEnergy);
    const double cosPZ = cos((double)primaryZenith);
    const int topoIdx = (topology < GPU_NUM_TOPOLOGIES) ? topology : 0;

    // Store precomputed transcendentals
    events.log10Energy[tid] = log10E;
    events.cosZenith[tid] = cosZ;
    events.log10PrimaryEnergy[tid] = log10PE;
    events.cosPrimaryZenith[tid] = cosPZ;

    // DOM efficiency reference evaluations
    double cachedDOMEffConv = 0.0;
    double cachedDOMEffPrompt = 0.0;
    double cachedDOMEffAstro = 0.0;

    if (splines.hasSplines) {
        const GPUSplineTable* s;

        s = splines.domEffSplines[GPU_FLUX_CONV][topoIdx];
        if (s != nullptr) {
            cachedDOMEffConv = evaluateSpline3D(*s, log10E, cosZ, splines.domEffReference);
        }

        s = splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx];
        if (s != nullptr) {
            cachedDOMEffPrompt = evaluateSpline3D(*s, log10E, cosZ, splines.domEffReference);
        }

        s = splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx];
        if (s != nullptr) {
            cachedDOMEffAstro = evaluateSpline3D(*s, log10E, cosZ, splines.domEffReference);
        }
    }

    events.cachedDOMEffConv[tid] = cachedDOMEffConv;
    events.cachedDOMEffPrompt[tid] = cachedDOMEffPrompt;
    events.cachedDOMEffAstro[tid] = cachedDOMEffAstro;

    // Hole ice reference evaluations
    double cachedHoleIceConv = 0.0;
    double cachedHoleIcePrompt = 0.0;
    double cachedHoleIceAstro = 0.0;

    if (splines.hasSplines) {
        const GPUSplineTable* s;

        s = splines.holeIceSplines[GPU_FLUX_CONV][topoIdx];
        if (s != nullptr) {
            cachedHoleIceConv = evaluateSpline3D(*s, log10E, cosZ, splines.holeIceReference);
        }

        s = splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx];
        if (s != nullptr) {
            cachedHoleIcePrompt = evaluateSpline3D(*s, log10E, cosZ, splines.holeIceReference);
        }

        s = splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx];
        if (s != nullptr) {
            cachedHoleIceAstro = evaluateSpline3D(*s, log10E, cosZ, splines.holeIceReference);
        }
    }

    events.cachedHoleIceConv[tid] = cachedHoleIceConv;
    events.cachedHoleIcePrompt[tid] = cachedHoleIcePrompt;
    events.cachedHoleIceAstro[tid] = cachedHoleIceAstro;

    //--------------------------------------------------------------------------
    // Cache spline basis functions for dims 0 & 1 (only if validated)
    //--------------------------------------------------------------------------

    if (splines.basisCacheValid && splines.hasSplines) {
        // DOM efficiency basis cache: find first non-null spline for this topology
        const GPUSplineTable* domEffRef = nullptr;
        for (int f = 0; f < GPU_NUM_FLUX_COMPONENTS && domEffRef == nullptr; ++f) {
            domEffRef = splines.domEffSplines[f][topoIdx];
        }

        if (domEffRef != nullptr) {
            // Check bounds for dims 0 and 1
            bool oob = (log10E <= domEffRef->knots[0][0] ||
                        log10E > domEffRef->knots[0][domEffRef->nknots[0] - 1] ||
                        cosZ <= domEffRef->knots[1][0] ||
                        cosZ > domEffRef->knots[1][domEffRef->nknots[1] - 1]);

            if (oob) {
                events.cachedDOMEffSpan0[tid] = -1;  // OOB sentinel
                events.cachedDOMEffSpan1[tid] = -1;
                events.cachedDOMEffBasis00[tid] = 0.0f;
                events.cachedDOMEffBasis01[tid] = 0.0f;
                events.cachedDOMEffBasis02[tid] = 0.0f;
                events.cachedDOMEffBasis10[tid] = 0.0f;
                events.cachedDOMEffBasis11[tid] = 0.0f;
                events.cachedDOMEffBasis12[tid] = 0.0f;
            } else {
                int sp0 = findKnotSpan(log10E, domEffRef->knots[0], domEffRef->nknots[0], domEffRef->order[0]);
                int sp1 = findKnotSpan(cosZ, domEffRef->knots[1], domEffRef->nknots[1], domEffRef->order[1]);
                double b0[8], b1[8];
                evaluateBasis(log10E, domEffRef->knots[0], domEffRef->nknots[0], sp0, domEffRef->order[0], b0);
                evaluateBasis(cosZ, domEffRef->knots[1], domEffRef->nknots[1], sp1, domEffRef->order[1], b1);

                events.cachedDOMEffSpan0[tid] = sp0;
                events.cachedDOMEffSpan1[tid] = sp1;
                events.cachedDOMEffBasis00[tid] = static_cast<float>(b0[0]);
                events.cachedDOMEffBasis01[tid] = static_cast<float>(b0[1]);
                events.cachedDOMEffBasis02[tid] = static_cast<float>(b0[2]);
                events.cachedDOMEffBasis10[tid] = static_cast<float>(b1[0]);
                events.cachedDOMEffBasis11[tid] = static_cast<float>(b1[1]);
                events.cachedDOMEffBasis12[tid] = static_cast<float>(b1[2]);
            }
        } else {
            // No DOM eff spline for this topology — set sentinel
            events.cachedDOMEffSpan0[tid] = -1;
            events.cachedDOMEffSpan1[tid] = -1;
            events.cachedDOMEffBasis00[tid] = 0.0f;
            events.cachedDOMEffBasis01[tid] = 0.0f;
            events.cachedDOMEffBasis02[tid] = 0.0f;
            events.cachedDOMEffBasis10[tid] = 0.0f;
            events.cachedDOMEffBasis11[tid] = 0.0f;
            events.cachedDOMEffBasis12[tid] = 0.0f;
        }

        // Hole ice basis cache: find first non-null spline for this topology
        const GPUSplineTable* holeIceRef = nullptr;
        for (int f = 0; f < GPU_NUM_FLUX_COMPONENTS && holeIceRef == nullptr; ++f) {
            holeIceRef = splines.holeIceSplines[f][topoIdx];
        }

        if (holeIceRef != nullptr) {
            bool oob = (log10E <= holeIceRef->knots[0][0] ||
                        log10E > holeIceRef->knots[0][holeIceRef->nknots[0] - 1] ||
                        cosZ <= holeIceRef->knots[1][0] ||
                        cosZ > holeIceRef->knots[1][holeIceRef->nknots[1] - 1]);

            if (oob) {
                events.cachedHoleIceSpan0[tid] = -1;
                events.cachedHoleIceSpan1[tid] = -1;
                events.cachedHoleIceBasis00[tid] = 0.0f;
                events.cachedHoleIceBasis01[tid] = 0.0f;
                events.cachedHoleIceBasis02[tid] = 0.0f;
                events.cachedHoleIceBasis10[tid] = 0.0f;
                events.cachedHoleIceBasis11[tid] = 0.0f;
                events.cachedHoleIceBasis12[tid] = 0.0f;
            } else {
                int sp0 = findKnotSpan(log10E, holeIceRef->knots[0], holeIceRef->nknots[0], holeIceRef->order[0]);
                int sp1 = findKnotSpan(cosZ, holeIceRef->knots[1], holeIceRef->nknots[1], holeIceRef->order[1]);
                double b0[8], b1[8];
                evaluateBasis(log10E, holeIceRef->knots[0], holeIceRef->nknots[0], sp0, holeIceRef->order[0], b0);
                evaluateBasis(cosZ, holeIceRef->knots[1], holeIceRef->nknots[1], sp1, holeIceRef->order[1], b1);

                events.cachedHoleIceSpan0[tid] = sp0;
                events.cachedHoleIceSpan1[tid] = sp1;
                events.cachedHoleIceBasis00[tid] = static_cast<float>(b0[0]);
                events.cachedHoleIceBasis01[tid] = static_cast<float>(b0[1]);
                events.cachedHoleIceBasis02[tid] = static_cast<float>(b0[2]);
                events.cachedHoleIceBasis10[tid] = static_cast<float>(b1[0]);
                events.cachedHoleIceBasis11[tid] = static_cast<float>(b1[1]);
                events.cachedHoleIceBasis12[tid] = static_cast<float>(b1[2]);
            }
        } else {
            events.cachedHoleIceSpan0[tid] = -1;
            events.cachedHoleIceSpan1[tid] = -1;
            events.cachedHoleIceBasis00[tid] = 0.0f;
            events.cachedHoleIceBasis01[tid] = 0.0f;
            events.cachedHoleIceBasis02[tid] = 0.0f;
            events.cachedHoleIceBasis10[tid] = 0.0f;
            events.cachedHoleIceBasis11[tid] = 0.0f;
            events.cachedHoleIceBasis12[tid] = 0.0f;
        }
    }
}

/**
 * @brief Launch the reference spline precomputation kernel.
 *
 * Should be called once after buildSplineLookup() and before any
 * likelihood evaluations.
 */
void launchPrecomputeReferenceSplines(
    GPUEventDataSoA& events,
    const GPUSplineLookup& splines,
    int numEvents,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    precomputeReferenceSplinesKernel<<<gridSize, blockSize, 0, stream>>>(
        events, splines, numEvents
    );

    CUDA_CHECK_KERNEL();

    // Synchronize to ensure cached values are ready before any LLH evaluation
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace gpu
} // namespace gollumfit
