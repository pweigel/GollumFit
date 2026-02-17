/**
 * @file event_weighting_gradient.cu
 * @brief GPU kernel for computing weight gradients via adjoint method.
 *
 * Two-kernel split approach for high performance:
 *
 * 1. computeGradientIntermediatesKernel: Forward pass that evaluates all splines
 *    and stores per-event intermediate scalar values + spline derivatives into a
 *    temporary buffer. ~same cost as the forward kernel.
 *
 * 2. computeAnalyticGradientKernel: Reads cached intermediates and computes all
 *    38 analytic partial derivatives dw/dp_j using the product/chain rule
 *    structure of the weight formula. No spline evaluations, minimal register
 *    pressure (processes derivatives in groups of 8 to avoid stack spill).
 *
 * This avoids the fundamental bottleneck of both previous approaches:
 * - GPUDual<38>: 14.6 KB stack spill/thread, 12.5% occupancy, memory-bound
 * - Multi-pass GPUDual<8>: 5x redundant spline evaluations + data reads
 * - Analytic-in-one-kernel: local_grad[38] still spills, re-evaluates splines
 *
 * The split decouples the expensive spline work (done once) from the cheap
 * derivative arithmetic (done with no spill).
 */

#include "cuda/GPUEventData.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUSplineTable.h"
#include <cmath>
#include <cfloat>

namespace gollumfit {
namespace gpu {

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
constexpr double GRAD_LN_10 = 2.302585092994046;
constexpr double GRAD_LN_2 = 0.6931471805599453;

//==============================================================================
// Intermediate buffer layout (SoA, 26 doubles per event)
//==============================================================================

// Offsets into the flat intermediate buffer (stride = numEvents between fields)
enum IntermediateField {
    IF_W = 0,           // final weight value
    IF_CONV,            // conventional component
    IF_PROMPT,          // prompt component
    IF_ASTRO,           // astrophysical component
    IF_ICEGRAD,         // ice gradient weight product
    IF_ATM_ADU,         // 1 + ADU * cachedAtmDensity
    IF_ATM_KLU,         // 1 + KLU * cachedKaonLosses
    IF_CONV_BASE,       // conv / convFlux = convDOMEff * convHoleIce * convAtt * atm_wgt
    IF_CONV_FLUX_OK,    // 1.0 if convFlux was NOT clamped, 0.0 if clamped
    IF_TILT_WGT,        // power law tilt weight
    IF_LOG_RATIO,       // log2(primaryEnergy / medianEnergy) for tilt derivative
    IF_BELOW_MEDIAN,    // 1.0 if primaryEnergy <= medianEnergy, 0.0 otherwise
    IF_NEUANEU_SIGN,    // +1.0 for antiparticle, -1.0 for particle
    IF_DRATE_DOMEFF_CONV,   // d(log10 rate)/d(domEff) for conventional
    IF_DRATE_DOMEFF_PROMPT, // d(log10 rate)/d(domEff) for prompt
    IF_DRATE_DOMEFF_ASTRO,  // d(log10 rate)/d(domEff) for astrophysical
    IF_DRATE_HOLEICE_CONV,  // d(log10 rate)/d(holeIce) for conventional
    IF_DRATE_HOLEICE_PROMPT,
    IF_DRATE_HOLEICE_ASTRO,
    IF_DATT_CONV,       // d(attenuation)/d(scale) for conventional
    IF_DATT_PROMPT,
    IF_DATT_ASTRO,
    IF_CONV_ATT,        // attenuation value for conventional (for safe division)
    IF_PROMPT_ATT,
    IF_ASTRO_ATT,
    IF_NEUANEU_WGT,     // neutrino-antineutrino weight
    NUM_INTERMEDIATE_FIELDS  // = 26
};

//==============================================================================
// Kernel 1: Compute gradient intermediates (forward pass + derivatives cache)
//==============================================================================

__global__ void __launch_bounds__(256, 2)
computeGradientIntermediatesKernel(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    const GPUSplineLookup splines,
    double* __restrict__ intermediates,  // [NUM_INTERMEDIATE_FIELDS * numEvents]
    const int numEvents,
    const bool enableTotalNorm
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    __shared__ double s_medianEnergy;

    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        s_medianEnergy = exp2(s_params[GP_ASTRO_PIVOT] * GRAD_LOG2_10);
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Helper macro for writing to SoA intermediate buffer
    #define WRITE_IF(field, val) intermediates[(field) * numEvents + tid] = (val)

    //--------------------------------------------------------------------------
    // Load event data (same as forward kernel)
    //--------------------------------------------------------------------------
    const float energy = events.energy[tid];
    const float zenith = events.zenith[tid];
    const float primaryEnergy = events.primaryEnergy[tid];
    const float primaryZenith = events.primaryZenith[tid];
    const int32_t primaryType = events.primaryType[tid];
    const uint32_t topology = events.topology[tid];

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

    const int32_t binIndex = events.binIndex[tid];

    // For events not in any bin, write zeros and return early
    if (binIndex < 0) {
        for (int f = 0; f < NUM_INTERMEDIATE_FIELDS; f++) {
            WRITE_IF(f, 0.0);
        }
        return;
    }

    //--------------------------------------------------------------------------
    // Spline corrections with derivatives
    //--------------------------------------------------------------------------
    const double log10Energy = log10((double)energy);
    const double cosZenith = cos((double)zenith);
    const double domEfficiency = s_params[GP_DELTA_DOMEFF];
    const double holeiceForward = s_params[GP_HOLEICE_FWD];
    const int topoIdx = (topology < GPU_NUM_TOPOLOGIES) ? topology : 0;

    // DOM efficiency corrections + z-derivatives
    double convDOMEff = 1.0, promptDOMEff = 1.0, astroDOMEff = 1.0;
    double drate_domeff_conv = 0.0, drate_domeff_prompt = 0.0, drate_domeff_astro = 0.0;

    if (splines.hasSplines) {
        // Helper lambda-like pattern for DOM efficiency
        auto evalDOMEff = [&](const GPUSplineTable* sp, double& correction, double& drate_dz) {
            correction = 1.0;
            drate_dz = 0.0;
            if (sp == nullptr) return;

            double rate, dz;
            evaluateSpline3DValueAndDerivZ(*sp, log10Energy, cosZenith, domEfficiency, rate, dz);
            if (rate == 0.0) { correction = 0.0; return; }

            double cache = evaluateSpline3D(*sp, log10Energy, cosZenith, splines.domEffReference);
            if (cache == 0.0 || cache < -1.0e30) { correction = 0.0; return; }

            double diff = rate - cache;
            correction = exp2(diff * GRAD_LOG2_10);
            drate_dz = dz;  // Store the raw d(log10_rate)/d(domEff)
        };

        evalDOMEff(splines.domEffSplines[GPU_FLUX_CONV][topoIdx], convDOMEff, drate_domeff_conv);
        evalDOMEff(splines.domEffSplines[GPU_FLUX_PROMPT][topoIdx], promptDOMEff, drate_domeff_prompt);
        evalDOMEff(splines.domEffSplines[GPU_FLUX_ASTRO][topoIdx], astroDOMEff, drate_domeff_astro);
    }

    // Hole ice corrections + z-derivatives
    double convHoleIce = 1.0, promptHoleIce = 1.0, astroHoleIce = 1.0;
    double drate_holeice_conv = 0.0, drate_holeice_prompt = 0.0, drate_holeice_astro = 0.0;

    if (splines.hasSplines) {
        auto evalHoleIce = [&](const GPUSplineTable* sp, double& correction, double& drate_dz) {
            correction = 1.0;
            drate_dz = 0.0;
            if (sp == nullptr) return;

            double rate, dz;
            evaluateSpline3DValueAndDerivZ(*sp, log10Energy, cosZenith, holeiceForward, rate, dz);
            if (rate == 0.0) { correction = 0.0; return; }

            double cache = evaluateSpline3D(*sp, log10Energy, cosZenith, splines.holeIceReference);
            if (cache == 0.0 || cache < -1.0e30) { correction = 0.0; return; }

            double diff = rate - cache;
            correction = exp2(diff * GRAD_LOG2_10);
            drate_dz = dz;
        };

        evalHoleIce(splines.holeIceSplines[GPU_FLUX_CONV][topoIdx], convHoleIce, drate_holeice_conv);
        evalHoleIce(splines.holeIceSplines[GPU_FLUX_PROMPT][topoIdx], promptHoleIce, drate_holeice_prompt);
        evalHoleIce(splines.holeIceSplines[GPU_FLUX_ASTRO][topoIdx], astroHoleIce, drate_holeice_astro);
    }

    // Attenuation corrections + z-derivatives
    double convAtt = 1.0, promptAtt = 1.0, astroAtt = 1.0;
    double datt_conv = 0.0, datt_prompt = 0.0, datt_astro = 0.0;

    if (splines.hasSplines) {
        const double cosPrimaryZenith = cos((double)primaryZenith);
        if (cosPrimaryZenith <= 0.1) {
            int ptypeIdx = mapParticleTypeToGPU(primaryType);
            if (ptypeIdx >= 0) {
                int xsParamIdx = (primaryType > 0) ? GP_NUXS : GP_NUBARXS;
                double scale = s_params[xsParamIdx];
                double log10PrimaryEnergy = log10((double)primaryEnergy);

                auto evalAtt = [&](const GPUSplineTable* sp, double& att_val, double& datt_dz) {
                    att_val = 1.0;
                    datt_dz = 0.0;
                    if (sp == nullptr) return;
                    evaluateSpline3DValueAndDerivZ(*sp, log10PrimaryEnergy, cosPrimaryZenith, scale, att_val, datt_dz);
                    if (att_val <= 0.0) { att_val = 0.0; datt_dz = 0.0; }
                };

                evalAtt(splines.attenSplines[GPU_FLUX_CONV][ptypeIdx], convAtt, datt_conv);
                evalAtt(splines.attenSplines[GPU_FLUX_PROMPT][ptypeIdx], promptAtt, datt_prompt);
                evalAtt(splines.attenSplines[GPU_FLUX_ASTRO][ptypeIdx], astroAtt, datt_astro);
            }
        }
    }

    //--------------------------------------------------------------------------
    // Compute weight components (same as forward kernel)
    //--------------------------------------------------------------------------
    // Ice gradient weight
    double icegrad_wgt = 1.0;
    icegrad_wgt *= fma(s_params[GP_ICEGRAD0], (double)cachedIceGrads[0], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD1], (double)cachedIceGrads[1], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD2], (double)cachedIceGrads[2], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD3], (double)cachedIceGrads[3], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD4], (double)cachedIceGrads[4], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD5], (double)cachedIceGrads[5], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD6], (double)cachedIceGrads[6], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD7], (double)cachedIceGrads[7], 1.0);
    icegrad_wgt *= fma(s_params[GP_ICEGRAD8], (double)cachedIceGrads[8], 1.0);

    // Atmospheric weight
    double atm_adu = fma(s_params[GP_ADU], (double)cachedAtmDensity, 1.0);
    double atm_klu = fma(s_params[GP_KLU], (double)cachedKaonLosses, 1.0);
    double atm_wgt = atm_adu * atm_klu;

    // Conventional flux
    double hadronic = 0.0;
    hadronic = fma(s_params[GP_HEKP], (double)events.cachedHadronicHEkp[tid], hadronic);
    hadronic = fma(s_params[GP_HEKM], (double)events.cachedHadronicHEkm[tid], hadronic);
    hadronic = fma(s_params[GP_VHE1PIP], (double)events.cachedHadronicVHE1pip[tid], hadronic);
    hadronic = fma(s_params[GP_VHE1PIM], (double)events.cachedHadronicVHE1pim[tid], hadronic);
    hadronic = fma(s_params[GP_VHE3KP], (double)events.cachedHadronicVHE3kp[tid], hadronic);
    hadronic = fma(s_params[GP_VHE3KM], (double)events.cachedHadronicVHE3km[tid], hadronic);
    hadronic = fma(s_params[GP_VHE3PIP], (double)events.cachedHadronicVHE3pip[tid], hadronic);
    hadronic = fma(s_params[GP_VHE3PIM], (double)events.cachedHadronicVHE3pim[tid], hadronic);
    hadronic = fma(s_params[GP_VHE3P], (double)events.cachedHadronicVHE3p[tid], hadronic);
    hadronic = fma(s_params[GP_VHE3N], (double)events.cachedHadronicVHE3n[tid], hadronic);

    double cr = 0.0;
    cr = fma(s_params[GP_CR1], (double)events.cachedCosmicRay1[tid], cr);
    cr = fma(s_params[GP_CR2], (double)events.cachedCosmicRay2[tid], cr);
    cr = fma(s_params[GP_CR3], (double)events.cachedCosmicRay3[tid], cr);
    cr = fma(s_params[GP_CR4], (double)events.cachedCosmicRay4[tid], cr);
    cr = fma(s_params[GP_CR5], (double)events.cachedCosmicRay5[tid], cr);
    cr = fma(s_params[GP_CR6], (double)events.cachedCosmicRay6[tid], cr);

    double convFlux = cachedConvWeight + cr + hadronic;
    double convFlux_ok = 1.0;
    if (convFlux < 0.0) {
        convFlux = (double)FLT_MAX;
        convFlux_ok = 0.0;
    }

    // Components
    double conv_base = convDOMEff * convHoleIce * convAtt * atm_wgt;
    double conv = conv_base * convFlux;
    double prompt = s_params[GP_PROMPT_NORM] * promptDOMEff * promptHoleIce *
                    promptAtt * cachedPromptWeight;

    // Astrophysical
    double neuaneu_sign = (primaryType < 0) ? 1.0 : -1.0;
    double neuaneu_wgt = (primaryType < 0) ? s_params[GP_NEUANEU_RATIO]
                                           : (2.0 - s_params[GP_NEUANEU_RATIO]);

    double ratio_val = (double)primaryEnergy / s_medianEnergy;
    double logRatio = log2(ratio_val);
    double belowMedian = ((double)primaryEnergy <= s_medianEnergy) ? 1.0 : 0.0;

    double deltaIndex = (belowMedian > 0.5) ? s_params[GP_ASTRO_DGAMMA]
                                             : s_params[GP_ASTRO_DGAMMA_SEC];
    double tilt_wgt = exp2(-deltaIndex * logRatio);

    double astro = s_params[GP_ASTRO_NORM] * astroDOMEff * astroHoleIce *
                   astroAtt * cachedAstroWeight * neuaneu_wgt * tilt_wgt;

    // Final weight
    double S = conv + prompt + astro;
    double w;
    if (enableTotalNorm) {
        w = s_params[GP_CONV_NORM] * S * icegrad_wgt;
    } else {
        w = (s_params[GP_CONV_NORM] * conv + prompt + astro) * icegrad_wgt;
    }

    //--------------------------------------------------------------------------
    // Write intermediates to SoA buffer
    //--------------------------------------------------------------------------
    WRITE_IF(IF_W, w);
    WRITE_IF(IF_CONV, conv);
    WRITE_IF(IF_PROMPT, prompt);
    WRITE_IF(IF_ASTRO, astro);
    WRITE_IF(IF_ICEGRAD, icegrad_wgt);
    WRITE_IF(IF_ATM_ADU, atm_adu);
    WRITE_IF(IF_ATM_KLU, atm_klu);
    WRITE_IF(IF_CONV_BASE, conv_base);
    WRITE_IF(IF_CONV_FLUX_OK, convFlux_ok);
    WRITE_IF(IF_TILT_WGT, tilt_wgt);
    WRITE_IF(IF_LOG_RATIO, logRatio);
    WRITE_IF(IF_BELOW_MEDIAN, belowMedian);
    WRITE_IF(IF_NEUANEU_SIGN, neuaneu_sign);
    WRITE_IF(IF_DRATE_DOMEFF_CONV, drate_domeff_conv);
    WRITE_IF(IF_DRATE_DOMEFF_PROMPT, drate_domeff_prompt);
    WRITE_IF(IF_DRATE_DOMEFF_ASTRO, drate_domeff_astro);
    WRITE_IF(IF_DRATE_HOLEICE_CONV, drate_holeice_conv);
    WRITE_IF(IF_DRATE_HOLEICE_PROMPT, drate_holeice_prompt);
    WRITE_IF(IF_DRATE_HOLEICE_ASTRO, drate_holeice_astro);
    WRITE_IF(IF_DATT_CONV, datt_conv);
    WRITE_IF(IF_DATT_PROMPT, datt_prompt);
    WRITE_IF(IF_DATT_ASTRO, datt_astro);
    WRITE_IF(IF_CONV_ATT, convAtt);
    WRITE_IF(IF_PROMPT_ATT, promptAtt);
    WRITE_IF(IF_ASTRO_ATT, astroAtt);
    WRITE_IF(IF_NEUANEU_WGT, neuaneu_wgt);

    #undef WRITE_IF
}

//==============================================================================
// Kernel 2: Analytic gradient from cached intermediates
//==============================================================================

__global__ void __launch_bounds__(256, 4)
computeAnalyticGradientKernel(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    const double* __restrict__ intermediates,  // [NUM_INTERMEDIATE_FIELDS * numEvents]
    const double* __restrict__ adjoint_wsum,
    const double* __restrict__ adjoint_w2sum,
    double* __restrict__ gradient,
    const int numEvents,
    const bool enableTotalNorm
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    __shared__ double s_grad[NUM_FIT_PARAMS];

    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
        s_grad[threadIdx.x] = 0.0;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Helper macro for reading from SoA intermediate buffer
    #define READ_IF(field) intermediates[(field) * numEvents + tid]

    // Per-thread gradient accumulator — process in groups of 8 to avoid spill.
    // We accumulate each group to shared memory before moving on.
    // But first, compute lambda and load intermediates for this event.

    double lambda = 0.0;
    double w = 0.0, conv = 0.0, prompt = 0.0, astro = 0.0;
    double icegrad = 0.0, conv_base = 0.0;
    double atm_adu = 0.0, atm_klu = 0.0;
    double convFlux_ok = 0.0;
    double tilt_wgt = 0.0, logRatio = 0.0, belowMedian = 0.0;
    double neuaneu_sign = 0.0, neuaneu_wgt = 0.0;
    double drate_domeff_conv = 0.0, drate_domeff_prompt = 0.0, drate_domeff_astro = 0.0;
    double drate_holeice_conv = 0.0, drate_holeice_prompt = 0.0, drate_holeice_astro = 0.0;
    double datt_conv = 0.0, datt_prompt = 0.0, datt_astro = 0.0;
    double convAtt = 0.0, promptAtt = 0.0, astroAtt = 0.0;

    bool active = false;

    if (tid < numEvents) {
        const int32_t binIndex = events.binIndex[tid];
        const int32_t numEventsInBin = events.numEvents[tid];

        if (binIndex >= 0) {
            // Load all intermediates
            w = READ_IF(IF_W);
            conv = READ_IF(IF_CONV);
            prompt = READ_IF(IF_PROMPT);
            astro = READ_IF(IF_ASTRO);
            icegrad = READ_IF(IF_ICEGRAD);
            atm_adu = READ_IF(IF_ATM_ADU);
            atm_klu = READ_IF(IF_ATM_KLU);
            conv_base = READ_IF(IF_CONV_BASE);
            convFlux_ok = READ_IF(IF_CONV_FLUX_OK);
            tilt_wgt = READ_IF(IF_TILT_WGT);
            logRatio = READ_IF(IF_LOG_RATIO);
            belowMedian = READ_IF(IF_BELOW_MEDIAN);
            neuaneu_sign = READ_IF(IF_NEUANEU_SIGN);
            drate_domeff_conv = READ_IF(IF_DRATE_DOMEFF_CONV);
            drate_domeff_prompt = READ_IF(IF_DRATE_DOMEFF_PROMPT);
            drate_domeff_astro = READ_IF(IF_DRATE_DOMEFF_ASTRO);
            drate_holeice_conv = READ_IF(IF_DRATE_HOLEICE_CONV);
            drate_holeice_prompt = READ_IF(IF_DRATE_HOLEICE_PROMPT);
            drate_holeice_astro = READ_IF(IF_DRATE_HOLEICE_ASTRO);
            datt_conv = READ_IF(IF_DATT_CONV);
            datt_prompt = READ_IF(IF_DATT_PROMPT);
            datt_astro = READ_IF(IF_DATT_ASTRO);
            convAtt = READ_IF(IF_CONV_ATT);
            promptAtt = READ_IF(IF_PROMPT_ATT);
            astroAtt = READ_IF(IF_ASTRO_ATT);
            neuaneu_wgt = READ_IF(IF_NEUANEU_WGT);

            // Compute lambda (adjoint contraction)
            double adj_ws = adjoint_wsum[binIndex];
            double adj_w2s = adjoint_w2sum[binIndex];
            double nEv = (numEventsInBin > 0) ? (double)numEventsInBin : 1.0;
            lambda = adj_ws + 2.0 * w / nEv * adj_w2s;
            active = true;
        }
    }

    #undef READ_IF

    //--------------------------------------------------------------------------
    // Compute derivatives in groups of 8, accumulating to shared memory
    // to keep per-thread register pressure minimal.
    //
    // All formulas assume enableTotalNorm:
    //   w = convNorm * S * icegrad,  S = conv + prompt + astro
    //   dw/dp = convNorm * icegrad * dS/dp  (for params in S)
    //
    // The factor  convNorm * icegrad  multiplies most derivatives through S.
    //--------------------------------------------------------------------------
    double CN = s_params[GP_CONV_NORM];
    double CN_IG = CN * icegrad;  // common factor for enableTotalNorm

    // NOTE: For !enableTotalNorm, the formula is different for conv-related params.
    // w = (CN*conv + prompt + astro) * icegrad
    // dw/d(conv_param) = CN * icegrad * d(conv)/dp
    // dw/d(prompt_param) = icegrad * d(prompt)/dp
    // dw/d(astro_param) = icegrad * d(astro)/dp
    // dw/d(CN) = conv * icegrad
    // The code below handles both cases.

    //--- Group 0: params 0-7 (convNorm, promptNorm, ADU, KLU, HEkp-VHE1pim) ---
    {
        double dw[8] = {0,0,0,0,0,0,0,0};

        if (active) {
            double S = conv + prompt + astro;

            // GP_CONV_NORM (0)
            if (enableTotalNorm)
                dw[0] = S * icegrad;
            else
                dw[0] = conv * icegrad;

            // GP_PROMPT_NORM (1): d(prompt)/d(pN) = prompt / promptNorm
            double pN = s_params[GP_PROMPT_NORM];
            double dprompt_dpN = (pN != 0.0) ? (prompt / pN) : 0.0;
            if (enableTotalNorm)
                dw[1] = CN_IG * dprompt_dpN;
            else
                dw[1] = icegrad * dprompt_dpN;

            // GP_ADU (2): d(conv)/d(ADU) = conv * cachedAtmDensity / atm_adu
            double cachedAtmDensity = (double)events.cachedAtmDensity[tid];
            double dconv_dADU = (atm_adu != 0.0) ? (conv * cachedAtmDensity / atm_adu) : 0.0;
            if (enableTotalNorm)
                dw[2] = CN_IG * dconv_dADU;
            else
                dw[2] = CN * icegrad * dconv_dADU;

            // GP_KLU (3): d(conv)/d(KLU) = conv * cachedKaonLosses / atm_klu
            double cachedKaonLosses = (double)events.cachedKaonLosses[tid];
            double dconv_dKLU = (atm_klu != 0.0) ? (conv * cachedKaonLosses / atm_klu) : 0.0;
            if (enableTotalNorm)
                dw[3] = CN_IG * dconv_dKLU;
            else
                dw[3] = CN * icegrad * dconv_dKLU;

            // GP_HEKP..VHE1PIM (4-7): d(conv)/dp_j = conv_base * cached_j (if not clamped)
            double flux_factor = conv_base * convFlux_ok;
            if (enableTotalNorm) {
                dw[4] = CN_IG * flux_factor * (double)events.cachedHadronicHEkp[tid];
                dw[5] = CN_IG * flux_factor * (double)events.cachedHadronicHEkm[tid];
                dw[6] = CN_IG * flux_factor * (double)events.cachedHadronicVHE1pip[tid];
                dw[7] = CN_IG * flux_factor * (double)events.cachedHadronicVHE1pim[tid];
            } else {
                double cn_ig = CN * icegrad;
                dw[4] = cn_ig * flux_factor * (double)events.cachedHadronicHEkp[tid];
                dw[5] = cn_ig * flux_factor * (double)events.cachedHadronicHEkm[tid];
                dw[6] = cn_ig * flux_factor * (double)events.cachedHadronicVHE1pip[tid];
                dw[7] = cn_ig * flux_factor * (double)events.cachedHadronicVHE1pim[tid];
            }
        }

        // Accumulate group to shared memory
        for (int j = 0; j < 8; j++) {
            atomicAdd(&s_grad[0 + j], lambda * dw[j]);
        }
    }

    __syncthreads();

    //--- Group 1: params 8-15 (VHE3kp-VHE3n, CR1, CR2) ---
    {
        double dw[8] = {0,0,0,0,0,0,0,0};

        if (active) {
            double flux_factor = conv_base * convFlux_ok;
            double outer = enableTotalNorm ? CN_IG : (CN * icegrad);

            dw[0] = outer * flux_factor * (double)events.cachedHadronicVHE3kp[tid];    // GP_VHE3KP (8)
            dw[1] = outer * flux_factor * (double)events.cachedHadronicVHE3km[tid];    // GP_VHE3KM (9)
            dw[2] = outer * flux_factor * (double)events.cachedHadronicVHE3pip[tid];   // GP_VHE3PIP (10)
            dw[3] = outer * flux_factor * (double)events.cachedHadronicVHE3pim[tid];   // GP_VHE3PIM (11)
            dw[4] = outer * flux_factor * (double)events.cachedHadronicVHE3p[tid];     // GP_VHE3P (12)
            dw[5] = outer * flux_factor * (double)events.cachedHadronicVHE3n[tid];     // GP_VHE3N (13)
            dw[6] = outer * flux_factor * (double)events.cachedCosmicRay1[tid];        // GP_CR1 (14)
            dw[7] = outer * flux_factor * (double)events.cachedCosmicRay2[tid];        // GP_CR2 (15)
        }

        for (int j = 0; j < 8; j++) {
            atomicAdd(&s_grad[8 + j], lambda * dw[j]);
        }
    }

    __syncthreads();

    //--- Group 2: params 16-23 (CR3-CR6, icegrad0-icegrad3) ---
    {
        double dw[8] = {0,0,0,0,0,0,0,0};

        if (active) {
            double flux_factor = conv_base * convFlux_ok;
            double outer = enableTotalNorm ? CN_IG : (CN * icegrad);

            // CR3-CR6 (16-19)
            dw[0] = outer * flux_factor * (double)events.cachedCosmicRay3[tid];
            dw[1] = outer * flux_factor * (double)events.cachedCosmicRay4[tid];
            dw[2] = outer * flux_factor * (double)events.cachedCosmicRay5[tid];
            dw[3] = outer * flux_factor * (double)events.cachedCosmicRay6[tid];

            // icegrad0-3 (20-23): dw/d(icegrad_k) = w * cachedIceGrad_k / (1 + p_k * c_k)
            for (int k = 0; k < 4; k++) {
                double c_k = (double)events.cachedIceGrad0[tid];  // Will be overridden below
                switch (k) {
                    case 0: c_k = (double)events.cachedIceGrad0[tid]; break;
                    case 1: c_k = (double)events.cachedIceGrad1[tid]; break;
                    case 2: c_k = (double)events.cachedIceGrad2[tid]; break;
                    case 3: c_k = (double)events.cachedIceGrad3[tid]; break;
                }
                double factor_k = fma(s_params[GP_ICEGRAD0 + k], c_k, 1.0);
                dw[4 + k] = (factor_k != 0.0) ? (w * c_k / factor_k) : 0.0;
            }
        }

        for (int j = 0; j < 8; j++) {
            atomicAdd(&s_grad[16 + j], lambda * dw[j]);
        }
    }

    __syncthreads();

    //--- Group 3: params 24-31 (icegrad4-8, domEff, holeIce, astroNorm) ---
    {
        double dw[8] = {0,0,0,0,0,0,0,0};

        if (active) {
            // icegrad4-8 (24-28)
            for (int k = 0; k < 5; k++) {
                double c_k = 0.0;
                switch (k) {
                    case 0: c_k = (double)events.cachedIceGrad4[tid]; break;
                    case 1: c_k = (double)events.cachedIceGrad5[tid]; break;
                    case 2: c_k = (double)events.cachedIceGrad6[tid]; break;
                    case 3: c_k = (double)events.cachedIceGrad7[tid]; break;
                    case 4: c_k = (double)events.cachedIceGrad8[tid]; break;
                }
                double factor_k = fma(s_params[GP_ICEGRAD4 + k], c_k, 1.0);
                dw[k] = (factor_k != 0.0) ? (w * c_k / factor_k) : 0.0;
            }

            // GP_DELTA_DOMEFF (29):
            // d(comp)/d(domEff) = comp * LN_10 * drate_domeff_comp
            // dS/d(domEff) = Σ d(comp)/d(domEff)
            double dconv_domeff = conv * GRAD_LN_10 * drate_domeff_conv;
            double dprompt_domeff = prompt * GRAD_LN_10 * drate_domeff_prompt;
            double dastro_domeff = astro * GRAD_LN_10 * drate_domeff_astro;
            if (enableTotalNorm)
                dw[5] = CN_IG * (dconv_domeff + dprompt_domeff + dastro_domeff);
            else
                dw[5] = icegrad * (CN * dconv_domeff + dprompt_domeff + dastro_domeff);

            // GP_HOLEICE_FWD (30): same structure as domEff
            double dconv_holeice = conv * GRAD_LN_10 * drate_holeice_conv;
            double dprompt_holeice = prompt * GRAD_LN_10 * drate_holeice_prompt;
            double dastro_holeice = astro * GRAD_LN_10 * drate_holeice_astro;
            if (enableTotalNorm)
                dw[6] = CN_IG * (dconv_holeice + dprompt_holeice + dastro_holeice);
            else
                dw[6] = icegrad * (CN * dconv_holeice + dprompt_holeice + dastro_holeice);

            // GP_ASTRO_NORM (31): d(astro)/d(astroNorm) = astro / astroNorm
            double aN = s_params[GP_ASTRO_NORM];
            double dastro_daN = (aN != 0.0) ? (astro / aN) : 0.0;
            if (enableTotalNorm)
                dw[7] = CN_IG * dastro_daN;
            else
                dw[7] = icegrad * dastro_daN;
        }

        for (int j = 0; j < 8; j++) {
            atomicAdd(&s_grad[24 + j], lambda * dw[j]);
        }
    }

    __syncthreads();

    //--- Group 4: params 32-37 (dGamma, dGammaSec, astroPivot, neuaneu, nuXS, nubarXS) ---
    {
        double dw[6] = {0,0,0,0,0,0};

        if (active) {
            // GP_ASTRO_DGAMMA (32): d(astro)/d(dgamma) = astro * (-LN_2) * logRatio  [if below median]
            double dastro_dgamma = (belowMedian > 0.5) ? (astro * (-GRAD_LN_2) * logRatio) : 0.0;
            if (enableTotalNorm)
                dw[0] = CN_IG * dastro_dgamma;
            else
                dw[0] = icegrad * dastro_dgamma;

            // GP_ASTRO_DGAMMA_SEC (33): same but above median
            double dastro_dgammaSec = (belowMedian < 0.5) ? (astro * (-GRAD_LN_2) * logRatio) : 0.0;
            if (enableTotalNorm)
                dw[1] = CN_IG * dastro_dgammaSec;
            else
                dw[1] = icegrad * dastro_dgammaSec;

            // GP_ASTRO_PIVOT (34):
            // logRatio = log2(E) - log2(E_med) = log2(E) - pivot * LOG2_10
            // d(logRatio)/d(pivot) = -LOG2_10
            // d(tilt_wgt)/d(pivot) = tilt_wgt * LN_2 * deltaIndex * LOG2_10
            //                      = tilt_wgt * LN_10 * deltaIndex
            // d(astro)/d(pivot) = astro * LN_10 * deltaIndex
            double deltaIndex = (belowMedian > 0.5) ? s_params[GP_ASTRO_DGAMMA]
                                                     : s_params[GP_ASTRO_DGAMMA_SEC];
            double dastro_dpivot = astro * GRAD_LN_10 * deltaIndex;
            if (enableTotalNorm)
                dw[2] = CN_IG * dastro_dpivot;
            else
                dw[2] = icegrad * dastro_dpivot;

            // GP_NEUANEU_RATIO (35):
            // d(neuaneu_wgt)/d(balance) = +1 (antiparticle) or -1 (particle)
            // d(astro)/d(balance) = (astro / neuaneu_wgt) * neuaneu_sign
            double dastro_dbal = (neuaneu_wgt != 0.0) ? (astro / neuaneu_wgt * neuaneu_sign) : 0.0;
            if (enableTotalNorm)
                dw[3] = CN_IG * dastro_dbal;
            else
                dw[3] = icegrad * dastro_dbal;

            // GP_NUXS (36) / GP_NUBARXS (37):
            // d(comp)/d(xs) = comp / att * datt_dz  (for the matching particle type)
            // Only the matching XS parameter gets a nonzero derivative
            bool isNu = (events.primaryType[tid] > 0);

            // Derivatives through attenuation
            double dconv_dxs = (convAtt != 0.0) ? (conv / convAtt * datt_conv) : 0.0;
            double dprompt_dxs = (promptAtt != 0.0) ? (prompt / promptAtt * datt_prompt) : 0.0;
            // For prompt, need to account for promptNorm already being in prompt
            // Actually prompt = pN * pDE * pHI * pAtt * cPW
            // d(prompt)/d(xs) = pN * pDE * pHI * datt_prompt * cPW = prompt / promptAtt * datt_prompt
            double dastro_dxs = (astroAtt != 0.0) ? (astro / astroAtt * datt_astro) : 0.0;

            double dS_dxs;
            if (enableTotalNorm)
                dS_dxs = CN_IG * (dconv_dxs + dprompt_dxs + dastro_dxs);
            else
                dS_dxs = icegrad * (CN * dconv_dxs + dprompt_dxs + dastro_dxs);

            if (isNu) {
                dw[4] = dS_dxs;  // GP_NUXS (36)
                dw[5] = 0.0;     // GP_NUBARXS (37)
            } else {
                dw[4] = 0.0;     // GP_NUXS (36)
                dw[5] = dS_dxs;  // GP_NUBARXS (37)
            }
        }

        for (int j = 0; j < 6; j++) {
            atomicAdd(&s_grad[32 + j], lambda * dw[j]);
        }
    }

    //--------------------------------------------------------------------------
    // Block-level reduction: write shared accumulator to global gradient
    //--------------------------------------------------------------------------
    __syncthreads();

    if (threadIdx.x < NUM_FIT_PARAMS && s_grad[threadIdx.x] != 0.0) {
        atomicAddDouble(&gradient[threadIdx.x], s_grad[threadIdx.x]);
    }
}

//==============================================================================
// Wrapper function (launches precompute + analytic gradient kernels)
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
    double* d_intermediates,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    // Zero gradient before computation
    CUDA_CHECK(cudaMemsetAsync(d_gradient, 0, NUM_FIT_PARAMS * sizeof(double), stream));

    int blockSize = 256;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    // Kernel 1: Forward pass + cache intermediates
    computeGradientIntermediatesKernel<<<gridSize, blockSize, 0, stream>>>(
        events, d_params, splines, d_intermediates, numEvents, enableTotalNorm);

    // Kernel 2: Compute analytic gradient from cached intermediates
    computeAnalyticGradientKernel<<<gridSize, blockSize, 0, stream>>>(
        events, d_params, d_intermediates, d_adjoint_wsum, d_adjoint_w2sum,
        d_gradient, numEvents, enableTotalNorm);

    CUDA_CHECK_KERNEL();
}

} // namespace gpu
} // namespace gollumfit
