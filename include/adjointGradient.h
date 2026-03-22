#ifndef ADJOINT_GRADIENT_H_INCLUDED
#define ADJOINT_GRADIENT_H_INCLUDED

#include <vector>
#include <deque>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <cstdint>

#include <PhysTools/autodiff.h>
#include <boost/math/special_functions/gamma.hpp>
#include <photospline/splinetable.h>

#include "GollumTools.h"

#include "Event.h"
#include "GollumEnumDefinitions.h"
#include "GollumParameters.h"

namespace gollumfit {
namespace adjoint {

// Spline map types (must match WeighterMaker definitions in analysisWeighting.h)
using DOMMapType = std::map<std::pair<FluxComponent, Topology>,
                            std::shared_ptr<photospline::splinetable<>>>;
using HoleIceMapType = std::map<std::pair<FluxComponent, Topology>,
                                std::shared_ptr<photospline::splinetable<>>>;
using AttenuationMapType = std::map<std::pair<FluxComponent, LW::ParticleType>,
                                    std::shared_ptr<photospline::splinetable<>>>;

constexpr double LN_10 = 2.302585092994046;
constexpr double LN_2 = 0.6931471805599453;
constexpr double LOG2_10 = 3.321928094887362;

// Per-event intermediate cache (26 fields)
struct EventCache {
    double w;               // final weight
    double conv;            // conventional component
    double prompt;          // prompt component
    double astro;           // astrophysical component
    double icegrad;         // ice gradient product
    double atm_adu;         // 1 + ADU * cachedAtmDensity
    double atm_klu;         // 1 + KLU * cachedKaonLosses
    double conv_base;       // conv without flux: DOMEff * HoleIce * Att * atm
    double convFlux_ok;     // 1.0 if flux >= 0, else 0.0
    double tilt_wgt;        // power-law tilt
    double logRatio;        // log2(primaryE / medianE)
    double belowMedian;     // 1.0 if below, 0.0 if above
    double neuaneu_sign;    // +1 antiparticle, -1 particle
    double drate_de_conv, drate_de_prompt, drate_de_astro;   // DOM eff spline z-deriv
    double drate_hi_conv, drate_hi_prompt, drate_hi_astro;   // hole ice spline z-deriv
    double datt_conv, datt_prompt, datt_astro;                // attenuation spline z-deriv
    double attConv, attPrompt, attAstro;                      // attenuation values
    double neuaneu_wgt;     // neutrino-antineutrino balance weight
};

//==============================================================================
// Helper: evaluate a 3D spline value and its z-derivative
//==============================================================================

inline std::pair<double, double> evalSplineWithZDeriv(
    const photospline::splinetable<>& spline, double coords[3])
{
    double val = spline(coords);
    int centers[3];
    if (!spline.searchcenters(coords, centers))
        return std::make_pair(val, 0.0);
    double dz = spline.ndsplineeval(coords, centers, 1 << 2);
    return std::make_pair(val, dz);
}

//==============================================================================
// Phase 1: Forward pass — scalar weights + cache intermediates + bin sums
//==============================================================================

inline double forwardPass(
    const std::vector<double>& params,
    const std::deque<Event>& events,
    const std::vector<int32_t>& binIndex,
    const std::vector<int32_t>& numEventsInBin,
    const DOMMapType& domEffMap,
    const HoleIceMapType& holeIceMap,
    const AttenuationMapType& attMap,
    bool enableTotalNorm,
    double sigmaOverMu,
    std::vector<EventCache>& cache,
    std::vector<double>& binSums,
    std::vector<double>& binSqSums,
    const std::vector<double>& dataCount,
    int numBins)
{
    // Unpack params
    const double convNorm   = params[kConvNorm];
    const double promptNorm = params[kPromptNorm];
    const double adu        = params[kZenithCorrection];
    const double klu        = params[kKaonLosses];
    const double hekp = params[kHadronicHEkp], hekm = params[kHadronicHEkm];
    const double vhe1pip = params[kHadronicVHE1pip], vhe1pim = params[kHadronicVHE1pim];
    const double vhe3kp = params[kHadronicVHE3kp], vhe3km = params[kHadronicVHE3km];
    const double vhe3pip = params[kHadronicVHE3pip], vhe3pim = params[kHadronicVHE3pim];
    const double vhe3p = params[kHadronicVHE3p], vhe3n = params[kHadronicVHE3n];
    const double cr1 = params[kCosmicRay1], cr2 = params[kCosmicRay2], cr3 = params[kCosmicRay3];
    const double cr4 = params[kCosmicRay4], cr5 = params[kCosmicRay5], cr6 = params[kCosmicRay6];
    const double ig[9] = {params[kIceGrad0], params[kIceGrad1], params[kIceGrad2],
                          params[kIceGrad3], params[kIceGrad4], params[kIceGrad5],
                          params[kIceGrad6], params[kIceGrad7], params[kIceGrad8]};
    const double deltaDomEff = params[kDomEfficiency];
    const double holeiceFwd  = params[kHoleiceForward];
    const double astroNorm   = params[kAstroNorm];
    const double dGamma      = params[kAstroDeltaGamma];
    const double dGammaSec   = params[kAstroDeltaGammaSec];
    const double astroPivot  = params[kAstroPivot];
    const double neuAneuBal  = params[kNeutrinoAntineutrinoRatio];
    const double nuxs        = params[kNuXS];
    const double nubarxs     = params[kNuBarXS];

    const double medianEnergy = std::pow(10.0, astroPivot);

    std::fill(binSums.begin(), binSums.end(), 0.0);
    std::fill(binSqSums.begin(), binSqSums.end(), 0.0);

    const size_t N = events.size();
    cache.resize(N);

    // Lambda: evaluate DOM eff or hole ice spline + z-derivative
    // Returns (correction_factor, z_derivative_of_log10_rate)
    auto evalDEorHI = [](const auto& splineMap, FluxComponent fc, Topology topo,
                         double log10E, double cosZ, double paramVal, double refVal)
        -> std::pair<double, double>
    {
        auto it = splineMap.find(std::make_pair(fc, topo));
        if (it == splineMap.end()) return std::make_pair(1.0, 0.0);
        double coords[3] = {log10E, cosZ, paramVal};
        auto [rate, dz] = evalSplineWithZDeriv(*it->second, coords);
        coords[2] = refVal;
        double ref = (*it->second)(coords);
        if (rate == 0.0 || ref == -std::numeric_limits<float>::max())
            return std::make_pair(0.0, 0.0);
        return std::make_pair(std::pow(10.0, rate - ref), dz);
    };

    // Lambda: evaluate attenuation spline + z-derivative
    // Only applies for upgoing events (cos(primaryZenith) <= 0.1)
    auto evalAtt = [](const AttenuationMapType& attMap, FluxComponent fc,
                      LW::ParticleType ptype, double log10PrimE, double cosPrimZ,
                      double xs) -> std::pair<double, double>
    {
        auto it = attMap.find(std::make_pair(fc, ptype));
        if (it == attMap.end()) return std::make_pair(1.0, 0.0);
        if (cosPrimZ > 0.1) return std::make_pair(1.0, 0.0);
        double coords[3] = {log10PrimE, cosPrimZ, xs};
        return evalSplineWithZDeriv(*it->second, coords);
    };

    for (size_t i = 0; i < N; i++) {
        const Event& e = events[i];
        EventCache& c = cache[i];

        if (binIndex[i] < 0) { c.w = 0.0; continue; }

        // Ice gradient weight
        const double cachedIG[9] = {
            e.cachedIceGrad0, e.cachedIceGrad1, e.cachedIceGrad2,
            e.cachedIceGrad3, e.cachedIceGrad4, e.cachedIceGrad5,
            e.cachedIceGrad6, e.cachedIceGrad7, e.cachedIceGrad8
        };
        double igWgt = 1.0;
        for (int k = 0; k < 9; k++) igWgt *= (1.0 + ig[k] * cachedIG[k]);

        // Atmospheric weight
        double atmAdu = 1.0 + adu * e.cachedAtmDensity;
        double atmKlu = 1.0 + klu * e.cachedKaonLosses;
        double atmWgt = atmAdu * atmKlu;

        // Conventional flux
        double convFlux = e.cachedConvWeight
            + hekp * e.cachedHadronicHEkp + hekm * e.cachedHadronicHEkm
            + vhe1pip * e.cachedHadronicVHE1pip + vhe1pim * e.cachedHadronicVHE1pim
            + vhe3kp * e.cachedHadronicVHE3kp + vhe3km * e.cachedHadronicVHE3km
            + vhe3pip * e.cachedHadronicVHE3pip + vhe3pim * e.cachedHadronicVHE3pim
            + vhe3p * e.cachedHadronicVHE3p + vhe3n * e.cachedHadronicVHE3n
            + cr1 * e.cachedCosmicRay1 + cr2 * e.cachedCosmicRay2
            + cr3 * e.cachedCosmicRay3 + cr4 * e.cachedCosmicRay4
            + cr5 * e.cachedCosmicRay5 + cr6 * e.cachedCosmicRay6;

        double fluxOk = 1.0;
        if (convFlux < 0.0) { convFlux = 0.0; fluxOk = 0.0; }

        // Spline evaluations
        auto topo = static_cast<Topology>(e.topology);
        double log10E = std::log10(e.energy);
        double cosZ = std::cos(e.zenith);

        auto [deConv, dz_de_conv]     = evalDEorHI(domEffMap, FluxComponent::atmConv,     topo, log10E, cosZ, deltaDomEff, 1.27);
        auto [dePrompt, dz_de_prompt] = evalDEorHI(domEffMap, FluxComponent::atmPrompt,   topo, log10E, cosZ, deltaDomEff, 1.27);
        auto [deAstro, dz_de_astro]   = evalDEorHI(domEffMap, FluxComponent::diffuseAstro, topo, log10E, cosZ, deltaDomEff, 1.27);

        auto [hiConv, dz_hi_conv]     = evalDEorHI(holeIceMap, FluxComponent::atmConv,     topo, log10E, cosZ, holeiceFwd, -1.0);
        auto [hiPrompt, dz_hi_prompt] = evalDEorHI(holeIceMap, FluxComponent::atmPrompt,   topo, log10E, cosZ, holeiceFwd, -1.0);
        auto [hiAstro, dz_hi_astro]   = evalDEorHI(holeIceMap, FluxComponent::diffuseAstro, topo, log10E, cosZ, holeiceFwd, -1.0);

        bool isNu = gollumfit::tools::isNeutrino(e.primaryType);
        double xs = isNu ? nuxs : nubarxs;
        double log10PrimE = std::log10(e.primaryEnergy);
        double cosPrimZ = std::cos(e.primaryZenith);

        auto [aConv, da_conv]     = evalAtt(attMap, FluxComponent::atmConv,     e.primaryType, log10PrimE, cosPrimZ, xs);
        auto [aPrompt, da_prompt] = evalAtt(attMap, FluxComponent::atmPrompt,   e.primaryType, log10PrimE, cosPrimZ, xs);
        auto [aAstro, da_astro]   = evalAtt(attMap, FluxComponent::diffuseAstro, e.primaryType, log10PrimE, cosPrimZ, xs);

        // Components
        double convBase = deConv * hiConv * aConv * atmWgt;
        double conv = convBase * convFlux;
        double prompt = promptNorm * dePrompt * hiPrompt * aPrompt * e.cachedPromptWeight;

        // Astro: tilt + neutrino-antineutrino balance
        double logRatio = std::log2(e.primaryEnergy) - astroPivot * LOG2_10;
        bool below = (e.primaryEnergy <= medianEnergy);
        double deltaIdx = below ? dGamma : dGammaSec;
        double tiltWgt = std::exp2(-deltaIdx * logRatio);

        // antiparticleWeighter: antineutrino -> balance, neutrino -> 2 - balance
        bool isAntiNu = ((int)e.primaryType < 0);
        double neuWgt = isAntiNu ? neuAneuBal : (2.0 - neuAneuBal);
        // d(neuWgt)/d(balance) = +1 for antineutrino, -1 for neutrino
        double neuSign = isAntiNu ? 1.0 : -1.0;

        double astro = astroNorm * deAstro * hiAstro * aAstro
                     * e.cachedAstroWeight * neuWgt * tiltWgt;

        // Final weight
        double w;
        if (enableTotalNorm)
            w = convNorm * (conv + prompt + astro) * igWgt;
        else
            w = (convNorm * conv + prompt + astro) * igWgt;

        // Cache
        c.w = w; c.conv = conv; c.prompt = prompt; c.astro = astro;
        c.icegrad = igWgt; c.atm_adu = atmAdu; c.atm_klu = atmKlu;
        c.conv_base = convBase; c.convFlux_ok = fluxOk;
        c.tilt_wgt = tiltWgt; c.logRatio = logRatio;
        c.belowMedian = below ? 1.0 : 0.0; c.neuaneu_sign = neuSign;
        c.drate_de_conv = dz_de_conv; c.drate_de_prompt = dz_de_prompt; c.drate_de_astro = dz_de_astro;
        c.drate_hi_conv = dz_hi_conv; c.drate_hi_prompt = dz_hi_prompt; c.drate_hi_astro = dz_hi_astro;
        c.datt_conv = da_conv; c.datt_prompt = da_prompt; c.datt_astro = da_astro;
        c.attConv = aConv; c.attPrompt = aPrompt; c.attAstro = aAstro;
        c.neuaneu_wgt = neuWgt;

        // Accumulate
        int bin = binIndex[i];
        binSums[bin] += w;
        int nEv = (e.num_events > 0) ? e.num_events : 1;
        binSqSums[bin] += (w * w) / nEv;
    }

    // SAY likelihood
    double nll = 0.0;
    for (int b = 0; b < numBins; b++) {
        double k = dataCount[b];
        double ws = binSums[b];
        double w2s = binSqSums[b];
        if (ws <= 0.0) continue;

        double s2 = w2s + sigmaOverMu * sigmaOverMu * ws * ws;
        if (s2 <= 0.0) {
            nll -= (k * std::log(ws) - ws - std::lgamma(k + 1.0));
            continue;
        }
        double alpha = ws * ws / s2 + 1.0;
        double beta = ws / s2;
        nll -= (alpha * std::log(beta) + std::lgamma(k + alpha)
                - std::lgamma(alpha) - std::lgamma(k + 1.0)
                - (k + alpha) * std::log1p(beta));
    }
    return nll;
}

//==============================================================================
// Phase 2: Bin adjoints — d(-logL)/d(w_sum) and d(-logL)/d(w2_sum)
//==============================================================================

inline void binAdjoints(
    const std::vector<double>& dataCount,
    const std::vector<double>& binSums,
    const std::vector<double>& binSqSums,
    double sigmaOverMu,
    std::vector<double>& adj_ws,
    std::vector<double>& adj_w2s,
    int numBins)
{
    using FD2 = phys_tools::autodiff::FD<2>;

    for (int b = 0; b < numBins; b++) {
        if (binSums[b] <= 0.0) { adj_ws[b] = 0.0; adj_w2s[b] = 0.0; continue; }

        double k = dataCount[b];
        FD2 ws(binSums[b], 0);
        FD2 w2s(binSqSums[b], 1);

        FD2 s2 = w2s + FD2(sigmaOverMu * sigmaOverMu) * ws * ws;
        FD2 llh;
        if (s2.value() <= 0.0) {
            llh = FD2(k) * log(ws) - ws - FD2(std::lgamma(k + 1.0));
        } else {
            FD2 alpha = ws * ws / s2 + FD2(1.0);
            FD2 beta = ws / s2;
            llh = alpha * log(beta)
                + lgamma(FD2(k) + alpha) - lgamma(alpha)
                - FD2(std::lgamma(k + 1.0))
                - (FD2(k) + alpha) * log(FD2(1.0) + beta);
        }
        adj_ws[b] = -llh.derivative(0);
        adj_w2s[b] = -llh.derivative(1);
    }
}

//==============================================================================
// Phase 3: Backward pass — analytic event gradients
//==============================================================================

inline void backwardPass(
    const std::vector<double>& params,
    const std::deque<Event>& events,
    const std::vector<EventCache>& cache,
    const std::vector<int32_t>& binIndex,
    const std::vector<int32_t>& numEventsInBin,
    const std::vector<double>& adj_ws,
    const std::vector<double>& adj_w2s,
    bool enableTotalNorm,
    std::vector<double>& grad)
{
    std::fill(grad.begin(), grad.end(), 0.0);
    const size_t N = events.size();

    for (size_t i = 0; i < N; i++) {
        const EventCache& c = cache[i];
        if (binIndex[i] < 0 || c.w == 0.0) continue;

        const Event& e = events[i];
        int bin = binIndex[i];
        int nEv = (e.num_events > 0) ? e.num_events : 1;
        double lam = adj_ws[bin] + 2.0 * c.w / nEv * adj_w2s[bin];

        double CN = params[kConvNorm];
        double S = c.conv + c.prompt + c.astro;

        // Common factors for enableTotalNorm vs not
        double CN_IG = CN * c.icegrad;
        double outer_conv  = enableTotalNorm ? CN_IG : (CN * c.icegrad);
        double outer_other = enableTotalNorm ? CN_IG : c.icegrad;

        // [0] convNorm
        grad[kConvNorm] += lam * (enableTotalNorm ? S * c.icegrad : c.conv * c.icegrad);

        // [1] promptNorm
        double pN = params[kPromptNorm];
        if (pN != 0.0) grad[kPromptNorm] += lam * outer_other * c.prompt / pN;

        // [2] ADU
        if (c.atm_adu != 0.0)
            grad[kZenithCorrection] += lam * outer_conv * c.conv * e.cachedAtmDensity / c.atm_adu;

        // [3] KLU
        if (c.atm_klu != 0.0)
            grad[kKaonLosses] += lam * outer_conv * c.conv * e.cachedKaonLosses / c.atm_klu;

        // [4-19] Hadronic + CR: dw/dp = outer_conv * conv_base * fluxOk * cached
        double ff = outer_conv * c.conv_base * c.convFlux_ok;
        grad[kHadronicHEkp]    += lam * ff * e.cachedHadronicHEkp;
        grad[kHadronicHEkm]    += lam * ff * e.cachedHadronicHEkm;
        grad[kHadronicVHE1pip] += lam * ff * e.cachedHadronicVHE1pip;
        grad[kHadronicVHE1pim] += lam * ff * e.cachedHadronicVHE1pim;
        grad[kHadronicVHE3kp]  += lam * ff * e.cachedHadronicVHE3kp;
        grad[kHadronicVHE3km]  += lam * ff * e.cachedHadronicVHE3km;
        grad[kHadronicVHE3pip] += lam * ff * e.cachedHadronicVHE3pip;
        grad[kHadronicVHE3pim] += lam * ff * e.cachedHadronicVHE3pim;
        grad[kHadronicVHE3p]   += lam * ff * e.cachedHadronicVHE3p;
        grad[kHadronicVHE3n]   += lam * ff * e.cachedHadronicVHE3n;
        grad[kCosmicRay1]     += lam * ff * e.cachedCosmicRay1;
        grad[kCosmicRay2]     += lam * ff * e.cachedCosmicRay2;
        grad[kCosmicRay3]     += lam * ff * e.cachedCosmicRay3;
        grad[kCosmicRay4]     += lam * ff * e.cachedCosmicRay4;
        grad[kCosmicRay5]     += lam * ff * e.cachedCosmicRay5;
        grad[kCosmicRay6]     += lam * ff * e.cachedCosmicRay6;

        // [20-28] Ice gradients: dw/dIG_k = w * cached_k / (1 + p_k * cached_k)
        const double cachedIG[9] = {
            e.cachedIceGrad0, e.cachedIceGrad1, e.cachedIceGrad2,
            e.cachedIceGrad3, e.cachedIceGrad4, e.cachedIceGrad5,
            e.cachedIceGrad6, e.cachedIceGrad7, e.cachedIceGrad8
        };
        for (int k = 0; k < 9; k++) {
            double denom = 1.0 + params[kIceGrad0 + k] * cachedIG[k];
            if (denom != 0.0) grad[kIceGrad0 + k] += lam * c.w * cachedIG[k] / denom;
        }

        // [29] DOM efficiency
        grad[kDomEfficiency] += lam * outer_other * LN_10 * (
            c.conv * c.drate_de_conv + c.prompt * c.drate_de_prompt + c.astro * c.drate_de_astro);

        // [30] Hole ice
        grad[kHoleiceForward] += lam * outer_other * LN_10 * (
            c.conv * c.drate_hi_conv + c.prompt * c.drate_hi_prompt + c.astro * c.drate_hi_astro);

        // [31] Astro norm
        double aN = params[kAstroNorm];
        if (aN != 0.0) grad[kAstroNorm] += lam * outer_other * c.astro / aN;

        // [32] deltaGamma (below median only)
        if (c.belowMedian > 0.5)
            grad[kAstroDeltaGamma] += lam * outer_other * c.astro * (-LN_2) * c.logRatio;

        // [33] deltaGammaSec (above median only)
        if (c.belowMedian < 0.5)
            grad[kAstroDeltaGammaSec] += lam * outer_other * c.astro * (-LN_2) * c.logRatio;

        // [34] astroPivot
        double deltaIdx = (c.belowMedian > 0.5) ? params[kAstroDeltaGamma] : params[kAstroDeltaGammaSec];
        grad[kAstroPivot] += lam * outer_other * c.astro * LN_10 * deltaIdx;

        // [35] neuAneuRatio
        if (c.neuaneu_wgt != 0.0)
            grad[kNeutrinoAntineutrinoRatio] += lam * outer_other * c.astro / c.neuaneu_wgt * c.neuaneu_sign;

        // [36-37] nuXS / nubarXS through attenuation
        double dc_xs = (c.attConv != 0.0) ? (c.conv / c.attConv * c.datt_conv) : 0.0;
        double dp_xs = (c.attPrompt != 0.0) ? (c.prompt / c.attPrompt * c.datt_prompt) : 0.0;
        double da_xs = (c.attAstro != 0.0) ? (c.astro / c.attAstro * c.datt_astro) : 0.0;
        double dS_xs;
        if (enableTotalNorm) dS_xs = CN_IG * (dc_xs + dp_xs + da_xs);
        else dS_xs = c.icegrad * (CN * dc_xs + dp_xs + da_xs);

        bool isNu = gollumfit::tools::isNeutrino(e.primaryType);
        if (isNu) grad[kNuXS] += lam * dS_xs;
        else grad[kNuBarXS] += lam * dS_xs;
    }
}

} // namespace adjoint
} // namespace gollumfit

#endif // ADJOINT_GRADIENT_H_INCLUDED
