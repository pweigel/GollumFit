/**
 * @file event_weighting_gradient.cu
 * @brief GPU kernel for computing weight gradients via adjoint method (step 3).
 *
 * For each event, computes dw_i/dp[j] for all 38 parameters using GPUDual<38>,
 * then contracts with per-bin adjoints to accumulate the gradient of -logL.
 *
 * Key optimization: spline evaluations are done as scalars with analytic z-derivatives,
 * then GPUDual<38> is constructed only for the final correction factor via chain rule.
 * This avoids propagating 38-component duals through binary search and basis evaluation.
 */

#include "cuda/GPUEventData.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUSplineTable.h"
#include <PhysTools/cuda/GPUAutodiff.h>
#include <cmath>
#include <cfloat>

namespace gollumfit {
namespace gpu {

using phys_tools::autodiff::gpu::GPUDual;
using Dual38 = GPUDual<38, double>;

//==============================================================================
// Parameter indices (must match event_weighting.cu)
//==============================================================================

enum GradParamIndex {
    GP_CONV_NORM = 0,
    GP_PROMPT_NORM = 1,
    GP_ADU = 2,
    GP_KLU = 3,
    GP_HEKP = 4,
    GP_HEKM = 5,
    GP_VHE1PIP = 6,
    GP_VHE1PIM = 7,
    GP_VHE3KP = 8,
    GP_VHE3KM = 9,
    GP_VHE3PIP = 10,
    GP_VHE3PIM = 11,
    GP_VHE3P = 12,
    GP_VHE3N = 13,
    GP_CR1 = 14,
    GP_CR2 = 15,
    GP_CR3 = 16,
    GP_CR4 = 17,
    GP_CR5 = 18,
    GP_CR6 = 19,
    GP_ICEGRAD0 = 20,
    GP_ICEGRAD1 = 21,
    GP_ICEGRAD2 = 22,
    GP_ICEGRAD3 = 23,
    GP_ICEGRAD4 = 24,
    GP_ICEGRAD5 = 25,
    GP_ICEGRAD6 = 26,
    GP_ICEGRAD7 = 27,
    GP_ICEGRAD8 = 28,
    GP_DELTA_DOMEFF = 29,
    GP_HOLEICE_FWD = 30,
    GP_ASTRO_NORM = 31,
    GP_ASTRO_DGAMMA = 32,
    GP_ASTRO_DGAMMA_SEC = 33,
    GP_ASTRO_PIVOT = 34,
    GP_NEUANEU_RATIO = 35,
    GP_NUXS = 36,
    GP_NUBARXS = 37
};

constexpr double GRAD_LOG2_10 = 3.321928094887362;

//==============================================================================
// Dual-number helper functions
//==============================================================================

/**
 * @brief Conventional flux with GPUDual<38>.
 * cachedConvWeight and hadronic/CR cached values are event constants (float→double).
 * Parameters are GPUDual<38>.
 */
__device__ Dual38 computeConvFluxDual(
    double cachedConvWeight,
    float cachedHEkp, float cachedHEkm,
    float cachedVHE1pip, float cachedVHE1pim,
    float cachedVHE3kp, float cachedVHE3km,
    float cachedVHE3pip, float cachedVHE3pim,
    float cachedVHE3p, float cachedVHE3n,
    float cachedCR1, float cachedCR2,
    float cachedCR3, float cachedCR4,
    float cachedCR5, float cachedCR6,
    const Dual38* p  // array of 38 dual params
) {
    // Start with base weight (constant)
    Dual38 flux(cachedConvWeight);

    // Hadronic: flux += param * cached  (param is dual, cached is scalar)
    flux += p[GP_HEKP] * (double)cachedHEkp;
    flux += p[GP_HEKM] * (double)cachedHEkm;
    flux += p[GP_VHE1PIP] * (double)cachedVHE1pip;
    flux += p[GP_VHE1PIM] * (double)cachedVHE1pim;
    flux += p[GP_VHE3KP] * (double)cachedVHE3kp;
    flux += p[GP_VHE3KM] * (double)cachedVHE3km;
    flux += p[GP_VHE3PIP] * (double)cachedVHE3pip;
    flux += p[GP_VHE3PIM] * (double)cachedVHE3pim;
    flux += p[GP_VHE3P] * (double)cachedVHE3p;
    flux += p[GP_VHE3N] * (double)cachedVHE3n;

    // Cosmic ray: flux += param * cached
    flux += p[GP_CR1] * (double)cachedCR1;
    flux += p[GP_CR2] * (double)cachedCR2;
    flux += p[GP_CR3] * (double)cachedCR3;
    flux += p[GP_CR4] * (double)cachedCR4;
    flux += p[GP_CR5] * (double)cachedCR5;
    flux += p[GP_CR6] * (double)cachedCR6;

    // Clamp negative flux (zero gradient through clamp)
    if (flux.value() < 0.0) {
        return Dual38((double)FLT_MAX);
    }

    return flux;
}

/**
 * @brief Ice gradient weight with GPUDual<38>.
 * Product of (1 + param_j * cached_j) for j=0..8.
 */
__device__ Dual38 computeIceGradWeightDual(
    const float* cachedIceGrads,
    const Dual38* p
) {
    Dual38 weight(1.0);
    // Multiply-accumulate chain to minimize live duals
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD0], (double)cachedIceGrads[0], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD1], (double)cachedIceGrads[1], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD2], (double)cachedIceGrads[2], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD3], (double)cachedIceGrads[3], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD4], (double)cachedIceGrads[4], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD5], (double)cachedIceGrads[5], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD6], (double)cachedIceGrads[6], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD7], (double)cachedIceGrads[7], 1.0);
    weight *= phys_tools::autodiff::gpu::fma(p[GP_ICEGRAD8], (double)cachedIceGrads[8], 1.0);
    return weight;
}

/**
 * @brief Antiparticle weight with GPUDual<38>.
 * Piecewise linear in P_NEUANEU_RATIO.
 */
__device__ __forceinline__ Dual38 computeAntiparticleWeightDual(
    int32_t primaryType,
    const Dual38& balance
) {
    if (primaryType < 0) {
        return balance;  // antiparticle: weight = balance
    } else {
        return Dual38(2.0) - balance;  // particle: weight = 2 - balance
    }
}

/**
 * @brief Power law tilt with GPUDual<38>.
 * Uses exp2/log2 from GPUAutodiff.h.
 *
 * medianEnergy = exp2(P_ASTRO_PIVOT * LOG2_10)  (this depends on P_ASTRO_PIVOT)
 * tilt = exp2(-deltaIndex * log2(energy / medianEnergy))
 *
 * We need the full chain through P_ASTRO_PIVOT, P_ASTRO_DGAMMA, P_ASTRO_DGAMMA_SEC.
 */
__device__ Dual38 computePowerLawTiltDual(
    float primaryEnergy,
    const Dual38& pivotParam,     // P_ASTRO_PIVOT
    const Dual38& dgamma,         // P_ASTRO_DGAMMA (below median)
    const Dual38& dgammaSec       // P_ASTRO_DGAMMA_SEC (above median)
) {
    using namespace phys_tools::autodiff::gpu;

    // medianEnergy = exp2(pivot * LOG2_10) = 10^pivot
    Dual38 medianEnergy = exp2(pivotParam * GRAD_LOG2_10);

    // logRatio = log2(primaryEnergy / medianEnergy)
    Dual38 ratio = Dual38((double)primaryEnergy) / medianEnergy;
    Dual38 logRatio = log2(ratio);

    // Branch on scalar comparison (same as forward pass)
    double medianVal = medianEnergy.value();
    if ((double)primaryEnergy > medianVal) {
        // Above median: tilt = exp2(-dgammaSec * logRatio)
        return exp2((Dual38(0.0) - dgammaSec) * logRatio);
    } else {
        // Below median: tilt = exp2(-dgamma * logRatio)
        return exp2((Dual38(0.0) - dgamma) * logRatio);
    }
}

/**
 * @brief DOM efficiency correction with analytic spline derivative + chain rule.
 *
 * correction = 10^(rate - cache) = exp2((rate - cache) * LOG2_10)
 * where rate = spline(log10E, cosZ, domEff), cache = spline(log10E, cosZ, ref)
 *
 * Since only the 3rd coord depends on P_DELTA_DOMEFF:
 *   d(correction)/d(domEff) = correction * LOG2_10 * ln(2) * d(rate)/d(domEff)
 *                            = correction * d(rate)/d(z)
 * (since LOG2_10 * ln(2) = log(10) * log2(e) * ln(2) ... let me be precise)
 *
 * Actually: correction = exp2((rate - cache) * LOG2_10)
 *   d(correction)/d(domEff) = correction * ln(2) * LOG2_10 * d(rate)/d(domEff)
 *                            = correction * log(10) * d(rate)/d(z)
 *
 * We construct a GPUDual<38> with value = correction, derivative[GP_DELTA_DOMEFF] set.
 */
__device__ Dual38 computeDOMEffCorrectionDual(
    const GPUSplineTable* spline,
    double log10Energy,
    double cosZenith,
    double domEfficiency,
    double referenceValue,
    int paramIndex  // GP_DELTA_DOMEFF
) {
    if (spline == nullptr) {
        return Dual38(1.0);
    }

    // Evaluate spline value and drate/d(domEff) at current domEff
    double rate, drate_dz;
    evaluateSpline3DValueAndDerivZ(*spline, log10Energy, cosZenith, domEfficiency, rate, drate_dz);

    if (rate == 0.0) {
        return Dual38(0.0);
    }

    // Evaluate at reference (scalar only, no derivative needed)
    double cache = evaluateSpline3D(*spline, log10Energy, cosZenith, referenceValue);
    if (cache == 0.0 || cache < -1.0e30) {
        return Dual38(0.0);
    }

    // correction = exp2((rate - cache) * LOG2_10) = 10^(rate - cache)
    double diff = rate - cache;
    double correction = exp2(diff * GRAD_LOG2_10);

    // d(correction)/d(domEff) = correction * ln(10) * drate/dz
    // ln(10) = LOG2_10 * ln(2) = 3.321928... * 0.693147... = 2.302585...
    constexpr double LN_10 = 2.302585092994046;
    double dcorr_ddomeff = correction * LN_10 * drate_dz;

    Dual38 result(correction);
    result.setDerivative(paramIndex, dcorr_ddomeff);
    return result;
}

/**
 * @brief Hole ice correction with analytic spline derivative + chain rule.
 * Same structure as DOM eff correction.
 */
__device__ Dual38 computeHoleIceCorrectionDual(
    const GPUSplineTable* spline,
    double log10Energy,
    double cosZenith,
    double holeiceForward,
    double referenceValue,
    int paramIndex  // GP_HOLEICE_FWD
) {
    if (spline == nullptr) {
        return Dual38(1.0);
    }

    double rate, drate_dz;
    evaluateSpline3DValueAndDerivZ(*spline, log10Energy, cosZenith, holeiceForward, rate, drate_dz);

    if (rate == 0.0) {
        return Dual38(0.0);
    }

    double cache = evaluateSpline3D(*spline, log10Energy, cosZenith, referenceValue);
    if (cache == 0.0 || cache < -1.0e30) {
        return Dual38(0.0);
    }

    double diff = rate - cache;
    double correction = exp2(diff * GRAD_LOG2_10);

    constexpr double LN_10 = 2.302585092994046;
    double dcorr_dholeice = correction * LN_10 * drate_dz;

    Dual38 result(correction);
    result.setDerivative(paramIndex, dcorr_dholeice);
    return result;
}

/**
 * @brief Attenuation correction with analytic spline derivative + chain rule.
 * Attenuation spline returns correction factor directly (no pow(10,x)).
 * The 3rd coordinate is P_NUXS or P_NUBARXS.
 */
__device__ Dual38 computeAttenuationCorrectionDual(
    const GPUSplineTable* spline,
    double log10PrimaryEnergy,
    double cosPrimaryZenith,
    double scale,
    int paramIndex  // GP_NUXS or GP_NUBARXS
) {
    if (spline == nullptr) {
        return Dual38(1.0);
    }

    double value, dvalue_dz;
    evaluateSpline3DValueAndDerivZ(*spline, log10PrimaryEnergy, cosPrimaryZenith, scale, value, dvalue_dz);

    // OOB coordinates → value=0 → correction=0 → weight=0 (matching CPU behavior).
    // CPU photospline returns 0 for OOB, and this zeros the event weight.
    if (value <= 0.0) {
        return Dual38(0.0);
    }

    Dual38 result(value);
    result.setDerivative(paramIndex, dvalue_dz);
    return result;
}

//==============================================================================
// Main gradient kernel
//==============================================================================

__global__ void __launch_bounds__(256, 1)
computeEventWeightGradientKernel(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    const GPUSplineLookup splines,
    const double* __restrict__ adjoint_wsum,
    const double* __restrict__ adjoint_w2sum,
    double* __restrict__ gradient,       // [NUM_FIT_PARAMS] global output
    const int numEvents,
    const bool enableTotalNorm
) {
    // Shared memory: parameters + block gradient accumulator
    __shared__ double s_params[NUM_FIT_PARAMS];
    __shared__ double s_grad[NUM_FIT_PARAMS];
    __shared__ double s_medianEnergy;

    // Load parameters to shared memory
    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
        s_grad[threadIdx.x] = 0.0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        s_medianEnergy = exp2(s_params[GP_ASTRO_PIVOT] * GRAD_LOG2_10);
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-thread local gradient accumulator
    double local_grad[NUM_FIT_PARAMS];
    #pragma unroll
    for (int j = 0; j < NUM_FIT_PARAMS; ++j) {
        local_grad[j] = 0.0;
    }

    if (tid < numEvents) {
        //----------------------------------------------------------------------
        // Load event data
        //----------------------------------------------------------------------
        const float energy = events.energy[tid];
        const float zenith = events.zenith[tid];
        const float primaryEnergy = events.primaryEnergy[tid];
        const float primaryZenith = events.primaryZenith[tid];
        const int32_t primaryType = events.primaryType[tid];
        const int32_t numEventsInBin = events.numEvents[tid];
        const uint32_t topology = events.topology[tid];
        const int32_t binIndex = events.binIndex[tid];

        const double cachedConvWeight = events.cachedConvWeight[tid];
        const double cachedPromptWeight = events.cachedPromptWeight[tid];
        const double cachedAstroWeight = events.cachedAstroWeight[tid];

        const float cachedAtmDensity = events.cachedAtmDensity[tid];
        const float cachedKaonLosses = events.cachedKaonLosses[tid];

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

        // Only compute gradient for events with valid bin assignments
        if (binIndex >= 0) {
            //------------------------------------------------------------------
            // Construct GPUDual<38> parameters on demand from shared memory
            //------------------------------------------------------------------
            Dual38 p[NUM_FIT_PARAMS];
            #pragma unroll
            for (int j = 0; j < NUM_FIT_PARAMS; ++j) {
                p[j] = Dual38(s_params[j], j);
            }

            //------------------------------------------------------------------
            // Spline-based corrections (scalar eval + chain rule)
            //------------------------------------------------------------------
            const double log10Energy = log10((double)energy);
            const double cosZenith = cos((double)zenith);
            const double domEfficiency = s_params[GP_DELTA_DOMEFF];
            const double holeiceForward = s_params[GP_HOLEICE_FWD];
            const int topoIdx = (topology < GPU_NUM_TOPOLOGIES) ? topology : 0;

            // DOM efficiency corrections
            Dual38 convDOMEff(1.0), promptDOMEff(1.0), astroDOMEff(1.0);
            if (splines.hasSplines) {
                convDOMEff = computeDOMEffCorrectionDual(
                    splines.domEffSplines[GPU_FLUX_CONV][topoIdx],
                    log10Energy, cosZenith, domEfficiency, splines.domEffReference, GP_DELTA_DOMEFF);
                promptDOMEff = computeDOMEffCorrectionDual(
                    splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx],
                    log10Energy, cosZenith, domEfficiency, splines.domEffReference, GP_DELTA_DOMEFF);
                astroDOMEff = computeDOMEffCorrectionDual(
                    splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx],
                    log10Energy, cosZenith, domEfficiency, splines.domEffReference, GP_DELTA_DOMEFF);
            }

            // Hole ice corrections
            Dual38 convHoleIce(1.0), promptHoleIce(1.0), astroHoleIce(1.0);
            if (splines.hasSplines) {
                convHoleIce = computeHoleIceCorrectionDual(
                    splines.holeIceSplines[GPU_FLUX_CONV][topoIdx],
                    log10Energy, cosZenith, holeiceForward, splines.holeIceReference, GP_HOLEICE_FWD);
                promptHoleIce = computeHoleIceCorrectionDual(
                    splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx],
                    log10Energy, cosZenith, holeiceForward, splines.holeIceReference, GP_HOLEICE_FWD);
                astroHoleIce = computeHoleIceCorrectionDual(
                    splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx],
                    log10Energy, cosZenith, holeiceForward, splines.holeIceReference, GP_HOLEICE_FWD);
            }

            // Attenuation corrections
            Dual38 convAtt(1.0), promptAtt(1.0), astroAtt(1.0);
            if (splines.hasSplines) {
                const double cosPrimaryZenith = cos((double)primaryZenith);
                if (cosPrimaryZenith <= 0.1) {
                    int ptypeIdx = mapParticleTypeToGPU(primaryType);
                    if (ptypeIdx >= 0) {
                        int xsParamIdx = (primaryType > 0) ? GP_NUXS : GP_NUBARXS;
                        double scale = s_params[xsParamIdx];
                        double log10PrimaryEnergy = log10((double)primaryEnergy);

                        convAtt = computeAttenuationCorrectionDual(
                            splines.attenSplines[GPU_FLUX_CONV][ptypeIdx],
                            log10PrimaryEnergy, cosPrimaryZenith, scale, xsParamIdx);
                        promptAtt = computeAttenuationCorrectionDual(
                            splines.attenSplines[GPU_FLUX_PROMPT][ptypeIdx],
                            log10PrimaryEnergy, cosPrimaryZenith, scale, xsParamIdx);
                        astroAtt = computeAttenuationCorrectionDual(
                            splines.attenSplines[GPU_FLUX_ASTRO][ptypeIdx],
                            log10PrimaryEnergy, cosPrimaryZenith, scale, xsParamIdx);
                    }
                }
            }

            //------------------------------------------------------------------
            // Compute weight components as GPUDual<38>
            //------------------------------------------------------------------

            // Ice gradient weight
            Dual38 icegrad_wgt = computeIceGradWeightDual(cachedIceGrads, p);

            // Atmospheric weight: (1 + ADU*cached) * (1 + KLU*cached)
            Dual38 atm_wgt = phys_tools::autodiff::gpu::fma(p[GP_ADU], (double)cachedAtmDensity, 1.0) *
                             phys_tools::autodiff::gpu::fma(p[GP_KLU], (double)cachedKaonLosses, 1.0);

            // Conventional flux
            Dual38 convFlux = computeConvFluxDual(
                cachedConvWeight,
                events.cachedHadronicHEkp[tid], events.cachedHadronicHEkm[tid],
                events.cachedHadronicVHE1pip[tid], events.cachedHadronicVHE1pim[tid],
                events.cachedHadronicVHE3kp[tid], events.cachedHadronicVHE3km[tid],
                events.cachedHadronicVHE3pip[tid], events.cachedHadronicVHE3pim[tid],
                events.cachedHadronicVHE3p[tid], events.cachedHadronicVHE3n[tid],
                events.cachedCosmicRay1[tid], events.cachedCosmicRay2[tid],
                events.cachedCosmicRay3[tid], events.cachedCosmicRay4[tid],
                events.cachedCosmicRay5[tid], events.cachedCosmicRay6[tid],
                p
            );

            // Conventional component
            Dual38 conv = convHoleIce * convDOMEff * convAtt * atm_wgt * convFlux;

            // Prompt component
            Dual38 prompt = p[GP_PROMPT_NORM] * promptHoleIce * promptDOMEff *
                            promptAtt * cachedPromptWeight;

            // Astrophysical component
            Dual38 neuaneu_wgt = computeAntiparticleWeightDual(primaryType, p[GP_NEUANEU_RATIO]);
            Dual38 tilt_wgt = computePowerLawTiltDual(
                primaryEnergy, p[GP_ASTRO_PIVOT], p[GP_ASTRO_DGAMMA], p[GP_ASTRO_DGAMMA_SEC]);
            Dual38 astro = p[GP_ASTRO_NORM] * astroHoleIce * astroDOMEff *
                           astroAtt * Dual38(cachedAstroWeight) * neuaneu_wgt * tilt_wgt;

            // Final weight
            Dual38 finalWeight;
            if (enableTotalNorm) {
                finalWeight = p[GP_CONV_NORM] * (conv + prompt + astro) * icegrad_wgt;
            } else {
                finalWeight = (p[GP_CONV_NORM] * conv + prompt + astro) * icegrad_wgt;
            }

            //------------------------------------------------------------------
            // Contract with bin adjoints
            //------------------------------------------------------------------
            double w_val = finalWeight.value();
            double adj_ws = adjoint_wsum[binIndex];
            double adj_w2s = adjoint_w2sum[binIndex];

            // w2_sum contribution: w^2/N, so dw2_i/dp = 2*w*dw/dp / N
            double nEv = (numEventsInBin > 0) ? (double)numEventsInBin : 1.0;
            double lambda = adj_ws + 2.0 * w_val / nEv * adj_w2s;

            // Accumulate gradient contribution: grad[j] += lambda * dw/dp[j]
            #pragma unroll
            for (int j = 0; j < NUM_FIT_PARAMS; ++j) {
                local_grad[j] += lambda * finalWeight.derivative(j);
            }
        }  // end if (binIndex >= 0)
    }  // end if (tid < numEvents)

    //--------------------------------------------------------------------------
    // Block-level reduction of gradients via shared memory
    //--------------------------------------------------------------------------
    __syncthreads();

    // Accumulate local gradients into shared memory
    // Process in chunks to fit in shared memory (reuse s_grad)
    for (int j = 0; j < NUM_FIT_PARAMS; ++j) {
        atomicAdd(&s_grad[j], local_grad[j]);
    }
    __syncthreads();

    // Block leader atomicAdds to global gradient
    if (threadIdx.x < NUM_FIT_PARAMS) {
        if (s_grad[threadIdx.x] != 0.0) {
            atomicAddDouble(&gradient[threadIdx.x], s_grad[threadIdx.x]);
        }
    }
}

//==============================================================================
// Wrapper function
//==============================================================================

void launchEventWeightGradientKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    const double* d_adjoint_wsum,
    const double* d_adjoint_w2sum,
    double* d_gradient,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    // Zero gradient before accumulation
    CUDA_CHECK(cudaMemsetAsync(d_gradient, 0, NUM_FIT_PARAMS * sizeof(double), stream));

    int blockSize = 256;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    computeEventWeightGradientKernel<<<gridSize, blockSize, 0, stream>>>(
        events, d_params, splines,
        d_adjoint_wsum, d_adjoint_w2sum,
        d_gradient, numEvents, enableTotalNorm
    );

    CUDA_CHECK_KERNEL();
}

} // namespace gpu
} // namespace gollumfit
