/**
 * @file event_weighting.cu
 * @brief CUDA kernel for computing event weights matching analysisWeighting.h logic.
 *
 * This file implements the GPU version of sterile::WeighterMaker::operator()
 * from analysisWeighting.h (lines 1100-1203). The kernel computes event weights
 * for each MC event given the 38 fit parameters.
 *
 * The weighting formula (from analysisWeighting.h:1197-1202):
 *   icegrad_wgt = (1+icegrad0*icegrad0_wgt)*...*(1+icegrad8*icegrad8_wgt)
 *   atm_wgt = (1+adu*adu_wgt)*(1+klu*klu_wgt)
 *   conv = convHoleIce * convDOMEff * convAtt * atm_wgt * ConvFluxWeighter(...)
 *   prompt = promptNorm * promptHoleIce * promptDOMEff * promptAtt * promptFlux
 *   astro = astroNorm * astroHoleIce * astroDOMEff * astroAtt * astroFlux * neuaneu_wgt * tiltWeighter
 *   weight = convNorm*(conv+prompt+astro)*icegrad_wgt (if enableTotalNorm)
 *         or (convNorm*conv+prompt+astro)*icegrad_wgt (otherwise)
 */

#include "cuda/GPUEventData.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUSplineTable.h"
// GPUAutodiff.h is part of PhysTools (located at PhysTools/PhysTools/cuda/)
#include <PhysTools/cuda/GPUAutodiff.h>
#include <cmath>
#include <cfloat>

namespace gollumfit {
namespace gpu {

//==============================================================================
// Parameter indices (matching analysisWeighting.h:1103-1141)
//==============================================================================

// These indices correspond to the parameter vector unpacking in WeighterMaker::operator()
enum ParamIndex {
    P_CONV_NORM = 0,
    P_PROMPT_NORM = 1,
    P_ADU = 2,              // Atmospheric density uncertainty
    P_KLU = 3,              // Kaon losses uncertainty
    P_HEKP = 4,             // High energy K+
    P_HEKM = 5,             // High energy K-
    P_VHE1PIP = 6,          // Very high energy pi+ (20 TeV)
    P_VHE1PIM = 7,          // Very high energy pi- (20 TeV)
    P_VHE3KP = 8,           // Very high energy K+ (2 PeV)
    P_VHE3KM = 9,           // Very high energy K- (2 PeV)
    P_VHE3PIP = 10,         // Very high energy pi+ (2 PeV)
    P_VHE3PIM = 11,         // Very high energy pi- (2 PeV)
    P_VHE3P = 12,           // Very high energy p (2 PeV)
    P_VHE3N = 13,           // Very high energy n (2 PeV)
    P_CR1 = 14,             // Cosmic ray parameter 1
    P_CR2 = 15,
    P_CR3 = 16,
    P_CR4 = 17,
    P_CR5 = 18,
    P_CR6 = 19,
    P_ICEGRAD0 = 20,        // Ice gradient parameters
    P_ICEGRAD1 = 21,
    P_ICEGRAD2 = 22,
    P_ICEGRAD3 = 23,
    P_ICEGRAD4 = 24,
    P_ICEGRAD5 = 25,
    P_ICEGRAD6 = 26,
    P_ICEGRAD7 = 27,
    P_ICEGRAD8 = 28,
    P_DELTA_DOMEFF = 29,    // DOM efficiency shift
    P_HOLEICE_FWD = 30,     // Hole ice forward parameter
    P_ASTRO_NORM = 31,      // Astrophysical normalization
    P_ASTRO_DGAMMA = 32,    // Spectral index variation
    P_ASTRO_DGAMMA_SEC = 33,// Secondary spectral index
    P_ASTRO_PIVOT = 34,     // Pivot energy for astro tilt
    P_NEUANEU_RATIO = 35,   // Neutrino/antineutrino ratio
    P_NUXS = 36,            // Neutrino cross section scale
    P_NUBARXS = 37          // Antineutrino cross section scale
};

//==============================================================================
// Device helper functions
//==============================================================================

/**
 * @brief Compute antiparticle weighting factor
 * Matches antiparticleWeighter from analysisWeighting.h:84-121
 */
__device__ __forceinline__ double computeAntiparticleWeight(
    int32_t primaryType,
    double balance
) {
    // Negative particle type indicates antiparticle
    return (primaryType < 0) ? balance : (2.0 - balance);
}

/**
 * @brief Compute broken power law tilt weighting (OPTIMIZED)
 * Matches brokenpowerlawTiltWeighter from analysisWeighting.h:136-175
 *
 * Optimization: Uses exp2/log2 instead of pow() for ~2.8x speedup.
 * The median energy should be precomputed in shared memory (once per block)
 * using: medianEnergy = exp2(medianLog10Energy * LOG2_10)
 *
 * Mathematical equivalence:
 *   pow(x, y) = 2^(y * log2(x)) = exp2(y * log2(x))
 *   pow(10, x) = 2^(x * log2(10)) = exp2(x * 3.321928...)
 */
__device__ __forceinline__ double computePowerLawTiltOptimized(
    float primaryEnergy,
    double precomputedMedianEnergy,  // Already computed: exp2(medianLog10Energy * LOG2_10)
    double deltaIndex1,              // Below median
    double deltaIndex2               // Above median
) {
    double ratio = primaryEnergy / precomputedMedianEnergy;
    double logRatio = log2(ratio);

    if (primaryEnergy > precomputedMedianEnergy) {
        return exp2(-deltaIndex2 * logRatio);
    } else {
        return exp2(-deltaIndex1 * logRatio);
    }
}

// Constant for pow(10, x) = exp2(x * LOG2_10)
constexpr double LOG2_10 = 3.321928094887362;

/**
 * @brief Compute broken power law tilt weighting (ORIGINAL - kept for reference)
 * Matches brokenpowerlawTiltWeighter from analysisWeighting.h:136-175
 */
__device__ __forceinline__ double computePowerLawTilt(
    float primaryEnergy,
    double medianLog10Energy,
    double deltaIndex1,     // Below median
    double deltaIndex2      // Above median
) {
    double medianEnergy = pow(10.0, medianLog10Energy);
    double ratio = primaryEnergy / medianEnergy;

    if (primaryEnergy > medianEnergy) {
        return pow(ratio, -deltaIndex2);
    } else {
        return pow(ratio, -deltaIndex1);
    }
}

/**
 * @brief Compute conventional flux contribution (MIXED PRECISION)
 * Matches ConvFluxWeighter from analysisWeighting.h:193-299
 *
 * Uses FP32 (float) for cached correction weights, FP64 for accumulation.
 * This provides ~1.65x speedup by reducing memory bandwidth.
 */
__device__ __forceinline__ double computeConvFlux(
    double cachedConvWeight,     // FP64 - base flux weight
    float cachedHadronicHEkp,    // FP32 - correction factor
    float cachedHadronicHEkm,
    float cachedHadronicVHE1pip,
    float cachedHadronicVHE1pim,
    float cachedHadronicVHE3kp,
    float cachedHadronicVHE3km,
    float cachedHadronicVHE3pip,
    float cachedHadronicVHE3pim,
    float cachedHadronicVHE3p,
    float cachedHadronicVHE3n,
    float cachedCR1,
    float cachedCR2,
    float cachedCR3,
    float cachedCR4,
    float cachedCR5,
    float cachedCR6,
    const double* __restrict__ params
) {
    // Hadronic contribution (FP32 inputs, FP64 accumulation with FMA)
    double hadronic = 0.0;
    hadronic = fma(params[P_HEKP], (double)cachedHadronicHEkp, hadronic);
    hadronic = fma(params[P_HEKM], (double)cachedHadronicHEkm, hadronic);
    hadronic = fma(params[P_VHE1PIP], (double)cachedHadronicVHE1pip, hadronic);
    hadronic = fma(params[P_VHE1PIM], (double)cachedHadronicVHE1pim, hadronic);
    hadronic = fma(params[P_VHE3KP], (double)cachedHadronicVHE3kp, hadronic);
    hadronic = fma(params[P_VHE3KM], (double)cachedHadronicVHE3km, hadronic);
    hadronic = fma(params[P_VHE3PIP], (double)cachedHadronicVHE3pip, hadronic);
    hadronic = fma(params[P_VHE3PIM], (double)cachedHadronicVHE3pim, hadronic);
    hadronic = fma(params[P_VHE3P], (double)cachedHadronicVHE3p, hadronic);
    hadronic = fma(params[P_VHE3N], (double)cachedHadronicVHE3n, hadronic);

    // Cosmic ray contribution (FP32 inputs, FP64 accumulation with FMA)
    double cr = 0.0;
    cr = fma(params[P_CR1], (double)cachedCR1, cr);
    cr = fma(params[P_CR2], (double)cachedCR2, cr);
    cr = fma(params[P_CR3], (double)cachedCR3, cr);
    cr = fma(params[P_CR4], (double)cachedCR4, cr);
    cr = fma(params[P_CR5], (double)cachedCR5, cr);
    cr = fma(params[P_CR6], (double)cachedCR6, cr);

    double flux = cachedConvWeight + cr + hadronic;

    // Return max float if negative (unphysical)
    if (flux < 0.0) {
        return static_cast<double>(FLT_MAX);
    }

    return flux;
}

/**
 * @brief Compute ice gradient weighting factor (MIXED PRECISION)
 * Product of (1 + param * cached_value) for all 9 ice gradient parameters
 * Uses FP32 inputs, FP64 accumulation with FMA
 */
__device__ __forceinline__ double computeIceGradWeight(
    const float* cachedIceGrads,  // FP32 cached values
    const double* __restrict__ params
) {
    double weight = 1.0;
    weight *= fma(params[P_ICEGRAD0], (double)cachedIceGrads[0], 1.0);
    weight *= fma(params[P_ICEGRAD1], (double)cachedIceGrads[1], 1.0);
    weight *= fma(params[P_ICEGRAD2], (double)cachedIceGrads[2], 1.0);
    weight *= fma(params[P_ICEGRAD3], (double)cachedIceGrads[3], 1.0);
    weight *= fma(params[P_ICEGRAD4], (double)cachedIceGrads[4], 1.0);
    weight *= fma(params[P_ICEGRAD5], (double)cachedIceGrads[5], 1.0);
    weight *= fma(params[P_ICEGRAD6], (double)cachedIceGrads[6], 1.0);
    weight *= fma(params[P_ICEGRAD7], (double)cachedIceGrads[7], 1.0);
    weight *= fma(params[P_ICEGRAD8], (double)cachedIceGrads[8], 1.0);
    return weight;
}

/**
 * @brief Compute DOM efficiency correction using spline evaluation
 *
 * Matches DOMEffWeighter from analysisWeighting.h:367-396:
 *   rate = spline(log10(energy), cos(zenith), domEfficiency)
 *   cache = spline(log10(energy), cos(zenith), 1.27)  // reference point
 *   correction = pow(10, rate - cache)
 *
 * NOTE: The CPU re-evaluates the spline at the reference point every time
 * (analysisWeighting.h:388-389) rather than using cached event fields.
 * The event fields (cachedDOMEffConv etc.) are initialized to 0 and never
 * populated from FastMC data, so we must match the CPU behavior.
 *
 * @param spline Pointer to the DOM efficiency spline for this flux/topology
 * @param log10Energy log10 of reconstructed energy
 * @param cosZenith cos(zenith angle)
 * @param domEfficiency Current DOM efficiency parameter value
 * @param referenceValue Reference DOM efficiency value (typically 1.27)
 * @return Correction factor, or 0.0 if out of parameter space, or 1.0 if no spline
 */
__device__ __forceinline__ double computeDOMEffCorrection(
    const GPUSplineTable* spline,
    double log10Energy,
    double cosZenith,
    double domEfficiency,
    double referenceValue
) {
    if (spline == nullptr) {
        return 1.0;
    }

    // Evaluate spline at current DOM efficiency
    double rate = evaluateSpline3D(*spline, log10Energy, cosZenith, domEfficiency);

    // Check for out-of-bounds (spline returns 0.0 for OOB)
    if (rate == 0.0) {
        return 0.0;  // Out of parameter space, zero event weight
    }

    // Evaluate spline at reference point (matching CPU: analysisWeighting.h:388-389)
    double cache = evaluateSpline3D(*spline, log10Energy, cosZenith, referenceValue);

    // Check for out-of-bounds at reference point
    if (cache == 0.0 || cache < -1.0e30) {
        return 0.0;  // Out of parameter space
    }

    // Compute correction: pow(10, rate - cache)
    return exp2((rate - cache) * LOG2_10);
}

/**
 * @brief Compute DOM efficiency correction using cached spline basis
 *
 * Uses precomputed span + basis for dims 0&1 and precomputed reference value
 * to skip redundant spline evaluations.
 *
 * @param spline Pointer to the DOM efficiency spline for this flux/topology
 * @param span0 Cached knot span for dim 0
 * @param span1 Cached knot span for dim 1
 * @param basis0 3 cached FP32 basis values for dim 0
 * @param basis1 3 cached FP32 basis values for dim 1
 * @param domEfficiency Current DOM efficiency parameter value
 * @param cachedRefValue Precomputed reference evaluation (from precomputeReferenceSplines)
 * @return Correction factor, or 0.0 if out of parameter space, or 1.0 if no spline
 */
__device__ __forceinline__ double computeDOMEffCorrectionCached(
    const GPUSplineTable* spline,
    int span0, int span1,
    const float* basis0, const float* basis1,
    double domEfficiency,
    double cachedRefValue
) {
    if (spline == nullptr) return 1.0;
    if (span0 < 0) return 0.0;  // OOB sentinel from precompute

    double rate = evaluateSpline3DCached(*spline, span0, span1, basis0, basis1, domEfficiency);
    if (rate == 0.0) return 0.0;
    if (cachedRefValue == 0.0) return 0.0;

    return exp2((rate - cachedRefValue) * LOG2_10);
}

/**
 * @brief Compute hole ice correction using cached spline basis
 *
 * Uses precomputed span + basis for dims 0&1 and precomputed reference value.
 *
 * @param spline Pointer to the hole ice spline for this flux/topology
 * @param span0 Cached knot span for dim 0
 * @param span1 Cached knot span for dim 1
 * @param basis0 3 cached FP32 basis values for dim 0
 * @param basis1 3 cached FP32 basis values for dim 1
 * @param holeiceForward Current hole ice forward parameter value
 * @param cachedRefValue Precomputed reference evaluation (from precomputeReferenceSplines)
 * @return Correction factor, or 0.0 if out of parameter space, or 1.0 if no spline
 */
__device__ __forceinline__ double computeHoleIceCorrectionCached(
    const GPUSplineTable* spline,
    int span0, int span1,
    const float* basis0, const float* basis1,
    double holeiceForward,
    double cachedRefValue
) {
    if (spline == nullptr) return 1.0;
    if (span0 < 0) return 0.0;  // OOB sentinel from precompute

    double rate = evaluateSpline3DCached(*spline, span0, span1, basis0, basis1, holeiceForward);
    if (rate == 0.0) return 0.0;
    if (cachedRefValue == 0.0) return 0.0;

    return exp2((rate - cachedRefValue) * LOG2_10);
}

/**
 * @brief Compute hole ice correction using spline evaluation
 *
 * Matches holeiceWeighter from analysisWeighting.h:640-672:
 *   rate = spline(log10(energy), cos(zenith), holeiceForward)
 *   cache = spline(log10(energy), cos(zenith), -1.0)  // reference point
 *   correction = pow(10, rate - cache)
 *
 * NOTE: Like DOM efficiency, the CPU re-evaluates at the reference point
 * every time rather than using cached event fields. We match that behavior.
 *
 * @param spline Pointer to the hole ice spline for this flux/topology
 * @param log10Energy log10 of reconstructed energy
 * @param cosZenith cos(zenith angle)
 * @param holeiceForward Current hole ice forward parameter value
 * @param referenceValue Reference hole ice value (typically -1.0)
 * @return Correction factor, or 0.0 if out of parameter space, or 1.0 if no spline
 */
__device__ __forceinline__ double computeHoleIceCorrection(
    const GPUSplineTable* spline,
    double log10Energy,
    double cosZenith,
    double holeiceForward,
    double referenceValue
) {
    if (spline == nullptr) {
        return 1.0;
    }

    // Evaluate spline at current hole ice parameter
    double rate = evaluateSpline3D(*spline, log10Energy, cosZenith, holeiceForward);

    // Check for out-of-bounds
    if (rate == 0.0) {
        return 0.0;  // Out of parameter space, zero event weight
    }

    // Evaluate spline at reference point (matching CPU behavior)
    double cache = evaluateSpline3D(*spline, log10Energy, cosZenith, referenceValue);

    // Check for out-of-bounds at reference point
    if (cache == 0.0 || cache < -1.0e30) {
        return 0.0;  // Out of parameter space
    }

    // Compute correction: pow(10, rate - cache)
    return exp2((rate - cache) * LOG2_10);
}

//==============================================================================
// Main Event Weighting Kernel (Value Only)
//==============================================================================

/**
 * @brief Compute event weights on GPU (matching WeighterMaker::operator())
 *
 * Each thread processes one event. Parameters are loaded to shared memory
 * for efficient access across the block.
 *
 * @param events GPU event data in SoA format
 * @param params 38 fit parameters
 * @param splines Spline lookup structure with DOM eff and hole ice splines
 * @param weights Output weights array [numEvents]
 * @param numEvents Total number of events
 * @param enableTotalNorm Steering parameter for normalization mode
 */
__global__ __launch_bounds__(256, 2) void computeEventWeightsKernel(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    const GPUSplineLookup splines,
    double* __restrict__ weights,
    const int numEvents,
    const bool enableTotalNorm
) {
    // Load parameters to shared memory (38 doubles = 304 bytes)
    __shared__ double s_params[NUM_FIT_PARAMS];
    // Precompute median energy for power law tilt (OPTIMIZATION: ~2.8x faster)
    __shared__ double s_medianEnergy;

    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    // Precompute median energy once per block using exp2/log2 optimization
    // pow(10, x) = exp2(x * log2(10))
    if (threadIdx.x == 0) {
        s_medianEnergy = exp2(s_params[P_ASTRO_PIVOT] * LOG2_10);
    }
    __syncthreads();

    // Global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    //--------------------------------------------------------------------------
    // Load event data (coalesced reads from SoA)
    //--------------------------------------------------------------------------

    const float primaryEnergy = events.primaryEnergy[tid];
    const int32_t primaryType = events.primaryType[tid];
    const uint32_t topology = events.topology[tid];

    // Flux weights
    const double cachedConvWeight = events.cachedConvWeight[tid];
    const double cachedPromptWeight = events.cachedPromptWeight[tid];
    const double cachedAstroWeight = events.cachedAstroWeight[tid];

    // Atmospheric weights (MIXED PRECISION: FP32)
    const float cachedAtmDensity = events.cachedAtmDensity[tid];
    const float cachedKaonLosses = events.cachedKaonLosses[tid];

    // Ice gradients (load all 9, MIXED PRECISION: FP32)
    float cachedIceGrads[9];
    cachedIceGrads[0] = events.cachedIceGrad0[tid];
    cachedIceGrads[1] = events.cachedIceGrad1[tid];
    cachedIceGrads[2] = events.cachedIceGrad2[tid];
    cachedIceGrads[3] = events.cachedIceGrad3[tid];
    cachedIceGrads[4] = events.cachedIceGrad4[tid];
    cachedIceGrads[5] = events.cachedIceGrad5[tid];
    cachedIceGrads[6] = events.cachedIceGrad6[tid];
    cachedIceGrads[7] = events.cachedIceGrad7[tid];
    cachedIceGrads[8] = events.cachedIceGrad8[tid];

    //--------------------------------------------------------------------------
    // Compute spline-based corrections
    //--------------------------------------------------------------------------

    // Use precomputed transcendentals (computed once at upload time)
    const double log10Energy = events.log10Energy[tid];
    const double cosZenith = events.cosZenith[tid];

    // Get current parameter values
    const double domEfficiency = s_params[P_DELTA_DOMEFF];
    const double holeiceForward = s_params[P_HOLEICE_FWD];

    // Topology index for spline lookup (0=cascade, 1=track)
    const int topoIdx = (topology < GPU_NUM_TOPOLOGIES) ? topology : 0;

    // Compute DOM efficiency corrections for each flux component
    double convDOMEff = 1.0;
    double promptDOMEff = 1.0;
    double astroDOMEff = 1.0;

    // Compute hole ice corrections for each flux component
    double convHoleIce = 1.0;
    double promptHoleIce = 1.0;
    double astroHoleIce = 1.0;

    if (splines.hasSplines && splines.basisCacheValid) {
        // Load cached spline basis for DOM eff
        const int domEffSpan0 = events.cachedDOMEffSpan0[tid];
        const int domEffSpan1 = events.cachedDOMEffSpan1[tid];
        float domEffBasis0[3] = {events.cachedDOMEffBasis00[tid], events.cachedDOMEffBasis01[tid], events.cachedDOMEffBasis02[tid]};
        float domEffBasis1[3] = {events.cachedDOMEffBasis10[tid], events.cachedDOMEffBasis11[tid], events.cachedDOMEffBasis12[tid]};

        // Load cached spline basis for hole ice
        const int holeIceSpan0 = events.cachedHoleIceSpan0[tid];
        const int holeIceSpan1 = events.cachedHoleIceSpan1[tid];
        float holeIceBasis0[3] = {events.cachedHoleIceBasis00[tid], events.cachedHoleIceBasis01[tid], events.cachedHoleIceBasis02[tid]};
        float holeIceBasis1[3] = {events.cachedHoleIceBasis10[tid], events.cachedHoleIceBasis11[tid], events.cachedHoleIceBasis12[tid]};

        // Load cached reference spline values
        const double cachedRefDOMEffConv = events.cachedDOMEffConv[tid];
        const double cachedRefDOMEffPrompt = events.cachedDOMEffPrompt[tid];
        const double cachedRefDOMEffAstro = events.cachedDOMEffAstro[tid];
        const double cachedRefHoleIceConv = events.cachedHoleIceConv[tid];
        const double cachedRefHoleIcePrompt = events.cachedHoleIcePrompt[tid];
        const double cachedRefHoleIceAstro = events.cachedHoleIceAstro[tid];

        convDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_CONV][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffConv);
        promptDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffPrompt);
        astroDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffAstro);

        convHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_CONV][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIceConv);
        promptHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIcePrompt);
        astroHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIceAstro);
    } else if (splines.hasSplines) {
        // Fallback: original full evaluation
        convDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_CONV][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);
        promptDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);
        astroDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);

        convHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_CONV][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
        promptHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
        astroHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
    }

    // Attenuation corrections
    // Matches attenuationWeighter from analysisWeighting.h:856-888
    // - Spline returns correction factor directly (no pow(10,x))
    // - Only applies for cos(primaryZenith) <= 0.1
    // - Scale parameter: nuxs for neutrinos, nubarxs for antineutrinos
    // - Only NuMu, NuMuBar, NuTau, NuTauBar have splines; others return 1.0
    double convAtt = 1.0;
    double promptAtt = 1.0;
    double astroAtt = 1.0;

    if (splines.hasSplines) {
        const double cosPrimaryZenith = events.cosPrimaryZenith[tid];
        if (cosPrimaryZenith <= 0.1) {
            int ptypeIdx = mapParticleTypeToGPU(primaryType);
            if (ptypeIdx >= 0) {
                // Select scale: neutrino (positive type) uses nuxs, anti uses nubarxs
                double scale = (primaryType > 0) ? s_params[P_NUXS] : s_params[P_NUBARXS];
                const double log10PrimaryEnergy = events.log10PrimaryEnergy[tid];

                const GPUSplineTable* convAttenSpline = splines.attenSplines[GPU_FLUX_CONV][ptypeIdx];
                if (convAttenSpline != nullptr) {
                    convAtt = evaluateSpline3D(*convAttenSpline, log10PrimaryEnergy, cosPrimaryZenith, scale);
                }

                const GPUSplineTable* promptAttenSpline = splines.attenSplines[GPU_FLUX_PROMPT][ptypeIdx];
                if (promptAttenSpline != nullptr) {
                    promptAtt = evaluateSpline3D(*promptAttenSpline, log10PrimaryEnergy, cosPrimaryZenith, scale);
                }

                const GPUSplineTable* astroAttenSpline = splines.attenSplines[GPU_FLUX_ASTRO][ptypeIdx];
                if (astroAttenSpline != nullptr) {
                    astroAtt = evaluateSpline3D(*astroAttenSpline, log10PrimaryEnergy, cosPrimaryZenith, scale);
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // Compute weights
    //--------------------------------------------------------------------------

    // Ice gradient weight (product of 9 terms)
    double icegrad_wgt = computeIceGradWeight(cachedIceGrads, s_params);

    // Atmospheric weight (MIXED PRECISION: FP32 inputs with FMA)
    double atm_wgt = fma(s_params[P_ADU], (double)cachedAtmDensity, 1.0) *
                     fma(s_params[P_KLU], (double)cachedKaonLosses, 1.0);

    // Conventional flux (includes hadronic and cosmic ray contributions)
    double convFlux = computeConvFlux(
        cachedConvWeight,
        events.cachedHadronicHEkp[tid],
        events.cachedHadronicHEkm[tid],
        events.cachedHadronicVHE1pip[tid],
        events.cachedHadronicVHE1pim[tid],
        events.cachedHadronicVHE3kp[tid],
        events.cachedHadronicVHE3km[tid],
        events.cachedHadronicVHE3pip[tid],
        events.cachedHadronicVHE3pim[tid],
        events.cachedHadronicVHE3p[tid],
        events.cachedHadronicVHE3n[tid],
        events.cachedCosmicRay1[tid],
        events.cachedCosmicRay2[tid],
        events.cachedCosmicRay3[tid],
        events.cachedCosmicRay4[tid],
        events.cachedCosmicRay5[tid],
        events.cachedCosmicRay6[tid],
        s_params
    );

    // Conventional component
    double conv = convHoleIce * convDOMEff * convAtt * atm_wgt * convFlux;

    // Prompt component
    double prompt = s_params[P_PROMPT_NORM] * promptHoleIce * promptDOMEff *
                    promptAtt * cachedPromptWeight;

    // Astrophysical component (using optimized power law tilt)
    double neuaneu_wgt = computeAntiparticleWeight(primaryType, s_params[P_NEUANEU_RATIO]);
    double tilt_wgt = computePowerLawTiltOptimized(
        primaryEnergy,
        s_medianEnergy,  // Precomputed in shared memory
        s_params[P_ASTRO_DGAMMA],
        s_params[P_ASTRO_DGAMMA_SEC]
    );
    double astro = s_params[P_ASTRO_NORM] * astroHoleIce * astroDOMEff *
                   astroAtt * cachedAstroWeight * neuaneu_wgt * tilt_wgt;

    //--------------------------------------------------------------------------
    // Final weight (matching analysisWeighting.h:1201-1202)
    //--------------------------------------------------------------------------

    double finalWeight;
    if (enableTotalNorm) {
        finalWeight = s_params[P_CONV_NORM] * (conv + prompt + astro) * icegrad_wgt;
    } else {
        finalWeight = (s_params[P_CONV_NORM] * conv + prompt + astro) * icegrad_wgt;
    }

    weights[tid] = finalWeight;
}

//==============================================================================
// Wrapper Function
//==============================================================================

/**
 * @brief Launch the event weighting kernel
 *
 * @param events GPU event data
 * @param d_params Device pointer to 38 parameters
 * @param splines Spline lookup structure
 * @param d_weights Device pointer to output weights
 * @param numEvents Number of events
 * @param enableTotalNorm Normalization mode flag
 * @param stream CUDA stream for async execution
 */
void launchEventWeightingKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    double* d_weights,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    computeEventWeightsKernel<<<gridSize, blockSize, 0, stream>>>(
        events, d_params, splines, d_weights, numEvents, enableTotalNorm
    );

    CUDA_CHECK_KERNEL();
}

//==============================================================================
// Squared Weight Kernel (for MC uncertainty estimation)
//==============================================================================

/**
 * @brief Compute squared event weights (for w^2 sum in SAY likelihood)
 *
 * This is a variant that computes w^2 / num_events for each event,
 * which is needed for the SAY likelihood MC uncertainty term.
 */
__global__ __launch_bounds__(256, 2) void computeEventWeightsSquaredKernel(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    const GPUSplineLookup splines,
    double* __restrict__ weightsSquared,
    const int numEvents,
    const bool enableTotalNorm
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    __shared__ double s_medianEnergy;  // Precomputed for optimized pow()

    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        s_medianEnergy = exp2(s_params[P_ASTRO_PIVOT] * LOG2_10);
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Load event data
    const float primaryEnergy = events.primaryEnergy[tid];
    const int32_t primaryType = events.primaryType[tid];
    const int32_t numEventsInBin = events.numEvents[tid];
    const uint32_t topology = events.topology[tid];

    const double cachedConvWeight = events.cachedConvWeight[tid];
    const double cachedPromptWeight = events.cachedPromptWeight[tid];
    const double cachedAstroWeight = events.cachedAstroWeight[tid];

    // Atmospheric weights (MIXED PRECISION: FP32)
    const float cachedAtmDensity = events.cachedAtmDensity[tid];
    const float cachedKaonLosses = events.cachedKaonLosses[tid];

    // Ice gradients (load all 9, MIXED PRECISION: FP32)
    float cachedIceGrads[9];
    cachedIceGrads[0] = events.cachedIceGrad0[tid];
    cachedIceGrads[1] = events.cachedIceGrad1[tid];
    cachedIceGrads[2] = events.cachedIceGrad2[tid];
    cachedIceGrads[3] = events.cachedIceGrad3[tid];
    cachedIceGrads[4] = events.cachedIceGrad4[tid];
    cachedIceGrads[5] = events.cachedIceGrad5[tid];
    cachedIceGrads[6] = events.cachedIceGrad6[tid];
    cachedIceGrads[7] = events.cachedIceGrad7[tid];
    cachedIceGrads[8] = events.cachedIceGrad8[tid];

    // Use precomputed transcendentals
    const double log10Energy = events.log10Energy[tid];
    const double cosZenith = events.cosZenith[tid];
    const double domEfficiency = s_params[P_DELTA_DOMEFF];
    const double holeiceForward = s_params[P_HOLEICE_FWD];
    const int topoIdx = (topology < GPU_NUM_TOPOLOGIES) ? topology : 0;

    double convDOMEff = 1.0, promptDOMEff = 1.0, astroDOMEff = 1.0;
    double convHoleIce = 1.0, promptHoleIce = 1.0, astroHoleIce = 1.0;

    if (splines.hasSplines && splines.basisCacheValid) {
        // Load cached spline basis for DOM eff
        const int domEffSpan0 = events.cachedDOMEffSpan0[tid];
        const int domEffSpan1 = events.cachedDOMEffSpan1[tid];
        float domEffBasis0[3] = {events.cachedDOMEffBasis00[tid], events.cachedDOMEffBasis01[tid], events.cachedDOMEffBasis02[tid]};
        float domEffBasis1[3] = {events.cachedDOMEffBasis10[tid], events.cachedDOMEffBasis11[tid], events.cachedDOMEffBasis12[tid]};

        // Load cached spline basis for hole ice
        const int holeIceSpan0 = events.cachedHoleIceSpan0[tid];
        const int holeIceSpan1 = events.cachedHoleIceSpan1[tid];
        float holeIceBasis0[3] = {events.cachedHoleIceBasis00[tid], events.cachedHoleIceBasis01[tid], events.cachedHoleIceBasis02[tid]};
        float holeIceBasis1[3] = {events.cachedHoleIceBasis10[tid], events.cachedHoleIceBasis11[tid], events.cachedHoleIceBasis12[tid]};

        // Load cached reference spline values
        const double cachedRefDOMEffConv = events.cachedDOMEffConv[tid];
        const double cachedRefDOMEffPrompt = events.cachedDOMEffPrompt[tid];
        const double cachedRefDOMEffAstro = events.cachedDOMEffAstro[tid];
        const double cachedRefHoleIceConv = events.cachedHoleIceConv[tid];
        const double cachedRefHoleIcePrompt = events.cachedHoleIcePrompt[tid];
        const double cachedRefHoleIceAstro = events.cachedHoleIceAstro[tid];

        convDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_CONV][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffConv);
        promptDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffPrompt);
        astroDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffAstro);

        convHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_CONV][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIceConv);
        promptHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIcePrompt);
        astroHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIceAstro);
    } else if (splines.hasSplines) {
        // Fallback: original full evaluation
        convDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_CONV][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);
        promptDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);
        astroDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);

        convHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_CONV][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
        promptHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
        astroHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
    }

    // Attenuation corrections
    double convAtt = 1.0, promptAtt = 1.0, astroAtt = 1.0;

    if (splines.hasSplines) {
        const double cosPrimaryZenith = events.cosPrimaryZenith[tid];
        if (cosPrimaryZenith <= 0.1) {
            int ptypeIdx = mapParticleTypeToGPU(primaryType);
            if (ptypeIdx >= 0) {
                double scale = (primaryType > 0) ? s_params[P_NUXS] : s_params[P_NUBARXS];
                const double log10PrimaryEnergy = events.log10PrimaryEnergy[tid];

                const GPUSplineTable* s;
                s = splines.attenSplines[GPU_FLUX_CONV][ptypeIdx];
                if (s != nullptr) { convAtt = evaluateSpline3D(*s, log10PrimaryEnergy, cosPrimaryZenith, scale); }
                s = splines.attenSplines[GPU_FLUX_PROMPT][ptypeIdx];
                if (s != nullptr) { promptAtt = evaluateSpline3D(*s, log10PrimaryEnergy, cosPrimaryZenith, scale); }
                s = splines.attenSplines[GPU_FLUX_ASTRO][ptypeIdx];
                if (s != nullptr) { astroAtt = evaluateSpline3D(*s, log10PrimaryEnergy, cosPrimaryZenith, scale); }
            }
        }
    }

    // Compute weights
    double icegrad_wgt = computeIceGradWeight(cachedIceGrads, s_params);

    double atm_wgt = fma(s_params[P_ADU], (double)cachedAtmDensity, 1.0) *
                     fma(s_params[P_KLU], (double)cachedKaonLosses, 1.0);

    double convFlux = computeConvFlux(
        cachedConvWeight,
        events.cachedHadronicHEkp[tid], events.cachedHadronicHEkm[tid],
        events.cachedHadronicVHE1pip[tid], events.cachedHadronicVHE1pim[tid],
        events.cachedHadronicVHE3kp[tid], events.cachedHadronicVHE3km[tid],
        events.cachedHadronicVHE3pip[tid], events.cachedHadronicVHE3pim[tid],
        events.cachedHadronicVHE3p[tid], events.cachedHadronicVHE3n[tid],
        events.cachedCosmicRay1[tid], events.cachedCosmicRay2[tid],
        events.cachedCosmicRay3[tid], events.cachedCosmicRay4[tid],
        events.cachedCosmicRay5[tid], events.cachedCosmicRay6[tid],
        s_params
    );

    double conv = convHoleIce * convDOMEff * convAtt * atm_wgt * convFlux;
    double prompt = s_params[P_PROMPT_NORM] * promptHoleIce * promptDOMEff * promptAtt * cachedPromptWeight;
    double neuaneu_wgt = computeAntiparticleWeight(primaryType, s_params[P_NEUANEU_RATIO]);
    double tilt_wgt = computePowerLawTiltOptimized(
        primaryEnergy, s_medianEnergy,
        s_params[P_ASTRO_DGAMMA], s_params[P_ASTRO_DGAMMA_SEC]
    );
    double astro = s_params[P_ASTRO_NORM] * astroHoleIce * astroDOMEff *
                   astroAtt * cachedAstroWeight * neuaneu_wgt * tilt_wgt;

    double weight;
    if (enableTotalNorm) {
        weight = s_params[P_CONV_NORM] * (conv + prompt + astro) * icegrad_wgt;
    } else {
        weight = (s_params[P_CONV_NORM] * conv + prompt + astro) * icegrad_wgt;
    }

    // Output w^2 / num_events (for SAY likelihood uncertainty)
    weightsSquared[tid] = (weight * weight) / static_cast<double>(numEventsInBin);
}

void launchEventWeightsSquaredKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    double* d_weightsSquared,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    computeEventWeightsSquaredKernel<<<gridSize, blockSize, 0, stream>>>(
        events, d_params, splines, d_weightsSquared, numEvents, enableTotalNorm
    );

    CUDA_CHECK_KERNEL();
}

//==============================================================================
// Combined Weights + Squared Weights Kernel (Optimized single-pass)
//==============================================================================

/**
 * @brief Compute both event weights and squared weights in a single kernel pass.
 *
 * This combined kernel is more efficient than running separate kernels as it:
 * - Loads parameters to shared memory once
 * - Loads all event data once
 * - Computes all intermediate values once
 * - Outputs both w and w^2/n in a single pass
 *
 * @param events GPU event data in SoA format
 * @param params 38 fit parameters
 * @param splines Spline lookup structure with DOM eff and hole ice splines
 * @param weights Output weights array [numEvents]
 * @param weightsSquared Output squared weights array [numEvents]
 * @param numEvents Total number of events
 * @param enableTotalNorm Steering parameter for normalization mode
 */
__global__ __launch_bounds__(256, 2) void computeEventWeightsWithSquaresKernel(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    const GPUSplineLookup splines,
    double* __restrict__ weights,
    double* __restrict__ weightsSquared,
    const int numEvents,
    const bool enableTotalNorm
) {
    // Load parameters to shared memory (38 doubles = 304 bytes)
    __shared__ double s_params[NUM_FIT_PARAMS];
    __shared__ double s_medianEnergy;  // Precomputed for optimized pow()

    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        s_medianEnergy = exp2(s_params[P_ASTRO_PIVOT] * LOG2_10);
    }
    __syncthreads();

    // Global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    //--------------------------------------------------------------------------
    // Load event data (coalesced reads from SoA)
    //--------------------------------------------------------------------------

    const float primaryEnergy = events.primaryEnergy[tid];
    const int32_t primaryType = events.primaryType[tid];
    const int32_t numEventsInBin = events.numEvents[tid];
    const uint32_t topology = events.topology[tid];

    // Flux weights
    const double cachedConvWeight = events.cachedConvWeight[tid];
    const double cachedPromptWeight = events.cachedPromptWeight[tid];
    const double cachedAstroWeight = events.cachedAstroWeight[tid];

    // Atmospheric weights (MIXED PRECISION: FP32)
    const float cachedAtmDensity = events.cachedAtmDensity[tid];
    const float cachedKaonLosses = events.cachedKaonLosses[tid];

    // Ice gradients (load all 9, MIXED PRECISION: FP32)
    float cachedIceGrads[9];
    cachedIceGrads[0] = events.cachedIceGrad0[tid];
    cachedIceGrads[1] = events.cachedIceGrad1[tid];
    cachedIceGrads[2] = events.cachedIceGrad2[tid];
    cachedIceGrads[3] = events.cachedIceGrad3[tid];
    cachedIceGrads[4] = events.cachedIceGrad4[tid];
    cachedIceGrads[5] = events.cachedIceGrad5[tid];
    cachedIceGrads[6] = events.cachedIceGrad6[tid];
    cachedIceGrads[7] = events.cachedIceGrad7[tid];
    cachedIceGrads[8] = events.cachedIceGrad8[tid];

    //--------------------------------------------------------------------------
    // Compute spline-based corrections
    //--------------------------------------------------------------------------

    // Use precomputed transcendentals (computed once at upload time)
    const double log10Energy = events.log10Energy[tid];
    const double cosZenith = events.cosZenith[tid];

    // Get current parameter values
    const double domEfficiency = s_params[P_DELTA_DOMEFF];
    const double holeiceForward = s_params[P_HOLEICE_FWD];

    // Topology index for spline lookup (0=cascade, 1=track)
    const int topoIdx = (topology < GPU_NUM_TOPOLOGIES) ? topology : 0;

    // Compute DOM efficiency corrections for each flux component
    double convDOMEff = 1.0;
    double promptDOMEff = 1.0;
    double astroDOMEff = 1.0;

    // Compute hole ice corrections for each flux component
    double convHoleIce = 1.0;
    double promptHoleIce = 1.0;
    double astroHoleIce = 1.0;

    if (splines.hasSplines && splines.basisCacheValid) {
        // Load cached spline basis for DOM eff
        const int domEffSpan0 = events.cachedDOMEffSpan0[tid];
        const int domEffSpan1 = events.cachedDOMEffSpan1[tid];
        float domEffBasis0[3] = {events.cachedDOMEffBasis00[tid], events.cachedDOMEffBasis01[tid], events.cachedDOMEffBasis02[tid]};
        float domEffBasis1[3] = {events.cachedDOMEffBasis10[tid], events.cachedDOMEffBasis11[tid], events.cachedDOMEffBasis12[tid]};

        // Load cached spline basis for hole ice
        const int holeIceSpan0 = events.cachedHoleIceSpan0[tid];
        const int holeIceSpan1 = events.cachedHoleIceSpan1[tid];
        float holeIceBasis0[3] = {events.cachedHoleIceBasis00[tid], events.cachedHoleIceBasis01[tid], events.cachedHoleIceBasis02[tid]};
        float holeIceBasis1[3] = {events.cachedHoleIceBasis10[tid], events.cachedHoleIceBasis11[tid], events.cachedHoleIceBasis12[tid]};

        // Load cached reference spline values
        const double cachedRefDOMEffConv = events.cachedDOMEffConv[tid];
        const double cachedRefDOMEffPrompt = events.cachedDOMEffPrompt[tid];
        const double cachedRefDOMEffAstro = events.cachedDOMEffAstro[tid];
        const double cachedRefHoleIceConv = events.cachedHoleIceConv[tid];
        const double cachedRefHoleIcePrompt = events.cachedHoleIcePrompt[tid];
        const double cachedRefHoleIceAstro = events.cachedHoleIceAstro[tid];

        convDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_CONV][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffConv);
        promptDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffPrompt);
        astroDOMEff = computeDOMEffCorrectionCached(
            splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx],
            domEffSpan0, domEffSpan1, domEffBasis0, domEffBasis1,
            domEfficiency, cachedRefDOMEffAstro);

        convHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_CONV][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIceConv);
        promptHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIcePrompt);
        astroHoleIce = computeHoleIceCorrectionCached(
            splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx],
            holeIceSpan0, holeIceSpan1, holeIceBasis0, holeIceBasis1,
            holeiceForward, cachedRefHoleIceAstro);
    } else if (splines.hasSplines) {
        // Fallback: original full evaluation
        convDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_CONV][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);
        promptDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);
        astroDOMEff = computeDOMEffCorrection(
            splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx],
            log10Energy, cosZenith, domEfficiency, splines.domEffReference);

        convHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_CONV][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
        promptHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
        astroHoleIce = computeHoleIceCorrection(
            splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx],
            log10Energy, cosZenith, holeiceForward, splines.holeIceReference);
    }

    // Attenuation corrections
    double convAtt = 1.0;
    double promptAtt = 1.0;
    double astroAtt = 1.0;

    if (splines.hasSplines) {
        const double cosPrimaryZenith = events.cosPrimaryZenith[tid];
        if (cosPrimaryZenith <= 0.1) {
            int ptypeIdx = mapParticleTypeToGPU(primaryType);
            if (ptypeIdx >= 0) {
                double scale = (primaryType > 0) ? s_params[P_NUXS] : s_params[P_NUBARXS];
                const double log10PrimaryEnergy = events.log10PrimaryEnergy[tid];

                const GPUSplineTable* s;
                s = splines.attenSplines[GPU_FLUX_CONV][ptypeIdx];
                if (s != nullptr) { convAtt = evaluateSpline3D(*s, log10PrimaryEnergy, cosPrimaryZenith, scale); }
                s = splines.attenSplines[GPU_FLUX_PROMPT][ptypeIdx];
                if (s != nullptr) { promptAtt = evaluateSpline3D(*s, log10PrimaryEnergy, cosPrimaryZenith, scale); }
                s = splines.attenSplines[GPU_FLUX_ASTRO][ptypeIdx];
                if (s != nullptr) { astroAtt = evaluateSpline3D(*s, log10PrimaryEnergy, cosPrimaryZenith, scale); }
            }
        }
    }

    //--------------------------------------------------------------------------
    // Compute weights
    //--------------------------------------------------------------------------

    // Ice gradient weight (product of 9 terms)
    double icegrad_wgt = computeIceGradWeight(cachedIceGrads, s_params);

    // Atmospheric weight (MIXED PRECISION: FP32 inputs with FMA)
    double atm_wgt = fma(s_params[P_ADU], (double)cachedAtmDensity, 1.0) *
                     fma(s_params[P_KLU], (double)cachedKaonLosses, 1.0);

    // Conventional flux (includes hadronic and cosmic ray contributions)
    double convFlux = computeConvFlux(
        cachedConvWeight,
        events.cachedHadronicHEkp[tid],
        events.cachedHadronicHEkm[tid],
        events.cachedHadronicVHE1pip[tid],
        events.cachedHadronicVHE1pim[tid],
        events.cachedHadronicVHE3kp[tid],
        events.cachedHadronicVHE3km[tid],
        events.cachedHadronicVHE3pip[tid],
        events.cachedHadronicVHE3pim[tid],
        events.cachedHadronicVHE3p[tid],
        events.cachedHadronicVHE3n[tid],
        events.cachedCosmicRay1[tid],
        events.cachedCosmicRay2[tid],
        events.cachedCosmicRay3[tid],
        events.cachedCosmicRay4[tid],
        events.cachedCosmicRay5[tid],
        events.cachedCosmicRay6[tid],
        s_params
    );

    // Conventional component
    double conv = convHoleIce * convDOMEff * convAtt * atm_wgt * convFlux;

    // Prompt component
    double prompt = s_params[P_PROMPT_NORM] * promptHoleIce * promptDOMEff *
                    promptAtt * cachedPromptWeight;

    // Astrophysical component (using optimized power law tilt)
    double neuaneu_wgt = computeAntiparticleWeight(primaryType, s_params[P_NEUANEU_RATIO]);
    double tilt_wgt = computePowerLawTiltOptimized(
        primaryEnergy,
        s_medianEnergy,  // Precomputed in shared memory
        s_params[P_ASTRO_DGAMMA],
        s_params[P_ASTRO_DGAMMA_SEC]
    );
    double astro = s_params[P_ASTRO_NORM] * astroHoleIce * astroDOMEff *
                   astroAtt * cachedAstroWeight * neuaneu_wgt * tilt_wgt;

    //--------------------------------------------------------------------------
    // Final weight (matching analysisWeighting.h:1201-1202)
    //--------------------------------------------------------------------------

    double finalWeight;
    if (enableTotalNorm) {
        finalWeight = s_params[P_CONV_NORM] * (conv + prompt + astro) * icegrad_wgt;
    } else {
        finalWeight = (s_params[P_CONV_NORM] * conv + prompt + astro) * icegrad_wgt;
    }

    // Output weight
    weights[tid] = finalWeight;

    // Output w^2 / num_events (for SAY likelihood uncertainty)
    // Handle edge case where numEventsInBin might be 0 or 1
    double nEv = (numEventsInBin > 0) ? static_cast<double>(numEventsInBin) : 1.0;
    weightsSquared[tid] = (finalWeight * finalWeight) / nEv;
}

/**
 * @brief Launch the combined weights + squared weights kernel
 */
void launchEventWeightingWithSquaresKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    double* d_weights,
    double* d_weightsSquared,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    computeEventWeightsWithSquaresKernel<<<gridSize, blockSize, 0, stream>>>(
        events, d_params, splines, d_weights, d_weightsSquared, numEvents, enableTotalNorm
    );

    CUDA_CHECK_KERNEL();
}

} // namespace gpu
} // namespace gollumfit
