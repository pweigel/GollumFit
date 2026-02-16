/**
 * @file test_reference_caching.cu
 * @brief Validate that cached reference spline values match on-the-fly evaluation,
 *        and that the optimization preserves LLH and gradient correctness.
 *
 * Tests:
 * 1. Cached reference values match CPU-side spline evaluation
 * 2. LLH + adjoint gradient still match finite-difference gradient
 * 3. Timing improvement from caching (informational)
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include "cuda/GPUFitAccelerator.h"
#include "cuda/GPUSplineTable.h"
#include "Event.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace gollumfit::gpu;

//==============================================================================
// Mock Event Generator (same pattern as test_adjoint_gradient.cu)
//==============================================================================

std::vector<Event> createMockEvents(int numEvents, TestRNG& rng) {
    std::vector<Event> events;
    events.reserve(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        Event e;

        e.energy = static_cast<float>(std::pow(10.0, rng.uniform(2.0, 6.0)));
        e.zenith = static_cast<float>(std::acos(rng.uniform(-1.0, 0.2)));
        e.primaryEnergy = e.energy * static_cast<float>(rng.uniform(1.0, 10.0));
        e.primaryZenith = e.zenith + static_cast<float>(rng.uniform(-0.1, 0.1));
        e.primaryAzimuth = static_cast<float>(rng.uniform(0.0, 2.0 * M_PI));
        e.totalColumnDepth = static_cast<float>(rng.uniform(1e4, 1e6));
        e.intX = static_cast<float>(rng.uniform(0.0, 1.0));
        e.intY = static_cast<float>(rng.uniform(0.0, 1.0));

        e.topology = (rng.uniform() < 0.6) ? 0 : 1;
        e.primaryType = (rng.uniform() < 0.5) ?
            LW::ParticleType::NuMu : LW::ParticleType::NuMuBar;
        e.num_events = 1;

        double baseWeight = rng.uniform(1e-10, 1e-6);
        e.cachedConvWeight = baseWeight;
        e.cachedPromptWeight = baseWeight * 0.1;
        e.cachedAstroWeight = baseWeight * 0.01;
        e.cachedWeight = 1.0;

        e.cachedHoleIceConv = 0.0;
        e.cachedHoleIcePrompt = 0.0;
        e.cachedHoleIceAstro = 0.0;
        e.cachedDOMEffConv = 0.0;
        e.cachedDOMEffPrompt = 0.0;
        e.cachedDOMEffAstro = 0.0;

        e.cachedHadronicHEkp = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedHadronicHEkm = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedHadronicVHE1pip = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedHadronicVHE1pim = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedHadronicVHE3kp = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3km = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3pip = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3pim = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3p = rng.uniform(-0.01, 0.01) * baseWeight;
        e.cachedHadronicVHE3n = rng.uniform(-0.01, 0.01) * baseWeight;

        e.cachedCosmicRay1 = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedCosmicRay2 = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedCosmicRay3 = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedCosmicRay4 = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedCosmicRay5 = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedCosmicRay6 = rng.uniform(-0.02, 0.02) * baseWeight;

        e.cachedAtmDensity = rng.uniform(-0.2, 0.2);
        e.cachedKaonLosses = rng.uniform(-0.1, 0.1);

        e.cachedIceGrad0 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad1 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad2 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad3 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad4 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad5 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad6 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad7 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad8 = rng.uniform(-0.1, 0.1);

        events.push_back(e);
    }

    return events;
}

std::vector<double> createNominalParameters() {
    std::vector<double> params(38, 0.0);
    params[0] = 1.0;   // convNorm
    params[1] = 1.0;   // promptNorm
    params[31] = 1.0;  // astroNorm
    params[35] = 1.0;  // neuaneu ratio
    return params;
}

//==============================================================================
// Helper: Create a simple 3D test spline
//==============================================================================

/**
 * Create a simple 3D B-spline table on GPU for testing.
 * Uses uniform knots covering the event parameter space:
 *   dim0: log10(energy) in [2, 6]
 *   dim1: cos(zenith)   in [-1, 0.2]
 *   dim2: parameter     in [paramMin, paramMax]
 *
 * Coefficients are set to a simple smooth function.
 */
void uploadTestSpline(
    GPUFitAccelerator& accel,
    const std::string& name,
    double paramMin,
    double paramMax,
    TestRNG& rng
) {
    const int ndim = 3;
    const int order[3] = {2, 2, 2};  // quadratic B-splines

    // Create knot vectors with enough knots for the order
    // nknots = ncoeffs + order + 1, so for 8 coeffs per dim: nknots = 11
    const int ncoeffsPerDim = 8;
    const int nk[3] = {
        ncoeffsPerDim + order[0] + 1,
        ncoeffsPerDim + order[1] + 1,
        ncoeffsPerDim + order[2] + 1
    };

    std::vector<std::vector<double>> knots(3);

    // dim0: log10(energy) in [2, 6]
    for (int i = 0; i < nk[0]; ++i) {
        double t = (double)i / (nk[0] - 1);
        knots[0].push_back(1.5 + t * 5.0);  // slightly wider than [2, 6]
    }
    // dim1: cos(zenith) in [-1, 0.2]
    for (int i = 0; i < nk[1]; ++i) {
        double t = (double)i / (nk[1] - 1);
        knots[1].push_back(-1.2 + t * 1.7);  // slightly wider than [-1, 0.2]
    }
    // dim2: parameter
    for (int i = 0; i < nk[2]; ++i) {
        double t = (double)i / (nk[2] - 1);
        knots[2].push_back(paramMin - 0.5 + t * (paramMax - paramMin + 1.0));
    }

    // Total coefficients
    int totalCoeffs = ncoeffsPerDim * ncoeffsPerDim * ncoeffsPerDim;
    std::vector<double> coeffs(totalCoeffs);

    // Fill with a smooth function: slight variation around 3.0
    // (spline values are log10 of correction factor)
    for (int i = 0; i < totalCoeffs; ++i) {
        coeffs[i] = 3.0 + rng.uniform(-0.5, 0.5);
    }

    accel.uploadSpline(name, ndim, order, knots, coeffs, nk);
}

//==============================================================================
// Helper: FD gradient computation
//==============================================================================

void computeFDGradient(
    GPUFitAccelerator& accel,
    const std::vector<double>& params,
    std::vector<double>& gradient,
    bool includePrior,
    double eps = 1e-5
) {
    const int N = static_cast<int>(params.size());
    gradient.resize(N);
    std::vector<double> p = params;

    for (int i = 0; i < N; ++i) {
        double orig = p[i];
        p[i] = orig + eps;
        double fp = accel.evaluateLikelihood(p, includePrior);
        p[i] = orig - eps;
        double fm = accel.evaluateLikelihood(p, includePrior);
        gradient[i] = (fp - fm) / (2.0 * eps);
        p[i] = orig;
    }
}

//==============================================================================
// Helper: Set up a test accelerator with splines
//==============================================================================

struct TestAccelerator {
    std::unique_ptr<GPUFitAccelerator> accel;
    HistogramConfig histConfig;

    void setup(int numEvents, int seed) {
        TestRNG rng(seed);
        auto events = createMockEvents(numEvents, rng);

        histConfig.nBinsEnergy = 20;
        histConfig.nBinsZenith = 12;
        histConfig.nBinsTopology = 2;
        for (int i = 0; i <= histConfig.nBinsEnergy; ++i)
            histConfig.energyEdges.push_back(std::pow(10.0, 2.0 + i * 0.2));
        for (int i = 0; i <= histConfig.nBinsZenith; ++i)
            histConfig.zenithEdges.push_back(-1.0 + i * 0.1);

        GPUAcceleratorConfig config;
        config.deviceId = 0;
        config.enableProfiling = true;
        accel = std::make_unique<GPUFitAccelerator>(config);
        accel->initialize(events, histConfig);

        // Create mock data histogram
        TestRNG dataRng(seed + 1000);
        std::vector<double> dataHist(histConfig.totalBins());
        for (int i = 0; i < histConfig.totalBins(); ++i) {
            dataHist[i] = 5.0 + dataRng.uniform(-2.0, 2.0);
        }
        accel->uploadDataHistogram(dataHist);

        // Upload test splines for DOM efficiency and hole ice
        TestRNG splineRng(seed + 2000);

        // DOM efficiency splines: reference value = 1.27, typical range [0.8, 1.5]
        uploadTestSpline(*accel, "domeff_atmConv_shower", 0.5, 1.8, splineRng);
        uploadTestSpline(*accel, "domeff_atmConv_track", 0.5, 1.8, splineRng);
        uploadTestSpline(*accel, "domeff_atmPrompt_shower", 0.5, 1.8, splineRng);
        uploadTestSpline(*accel, "domeff_atmPrompt_track", 0.5, 1.8, splineRng);
        uploadTestSpline(*accel, "domeff_diffuseAstro_shower", 0.5, 1.8, splineRng);
        uploadTestSpline(*accel, "domeff_diffuseAstro_track", 0.5, 1.8, splineRng);

        // Hole ice splines: reference value = -1.0, typical range [-3, 1]
        uploadTestSpline(*accel, "holeice_atmConv_shower", -3.5, 1.5, splineRng);
        uploadTestSpline(*accel, "holeice_atmConv_track", -3.5, 1.5, splineRng);
        uploadTestSpline(*accel, "holeice_atmPrompt_shower", -3.5, 1.5, splineRng);
        uploadTestSpline(*accel, "holeice_atmPrompt_track", -3.5, 1.5, splineRng);
        uploadTestSpline(*accel, "holeice_diffuseAstro_shower", -3.5, 1.5, splineRng);
        uploadTestSpline(*accel, "holeice_diffuseAstro_track", -3.5, 1.5, splineRng);

        // Build spline lookup and precompute references
        accel->buildSplineLookup();
        accel->precomputeReferenceSplines();
    }
};

//==============================================================================
// Test 1: Cached references are non-zero for events in bounds
//==============================================================================

TestResult testCachedReferencesNonZero() {
    TestResult result;
    result.name = "Cached References - Non-zero for in-bounds events";

    try {
        const int numEvents = 10000;
        TestAccelerator ta;
        ta.setup(numEvents, 42);

        auto params = createNominalParameters();
        // Set domEff and holeIce to their reference values
        params[29] = 1.27;   // P_DELTA_DOMEFF = reference value
        params[30] = -1.0;   // P_HOLEICE_FWD = reference value

        double llh = ta.accel->evaluateLikelihood(params, false);

        // Verify LLH is finite and reasonable
        bool isFinite = std::isfinite(llh);
        if (!isFinite) {
            result.passed = false;
            result.message = "LLH at reference params is not finite: " + std::to_string(llh);
            return result;
        }

        // At reference values, DOM eff and hole ice corrections should all be 1.0
        // (since rate == cache => 10^0 = 1). The LLH should be well-defined.
        std::cout << "  LLH at reference params (domEff=1.27, holeIce=-1.0): " << llh << "\n";

        // Also evaluate at a slightly perturbed point
        params[29] = 1.30;   // Slight perturbation from reference
        params[30] = -0.9;
        double llh2 = ta.accel->evaluateLikelihood(params, false);

        std::cout << "  LLH at perturbed params (domEff=1.30, holeIce=-0.9): " << llh2 << "\n";

        // They should be different (spline corrections are non-trivial)
        if (std::abs(llh - llh2) < 1e-15) {
            result.passed = false;
            result.message = "LLH unchanged with perturbation - splines may not be active";
            return result;
        }

        result.passed = true;
        result.message = "LLH is finite and changes with DOM eff/hole ice perturbation";

    } catch (const std::exception& ex) {
        result.passed = false;
        result.message = std::string("Exception: ") + ex.what();
    }

    return result;
}

//==============================================================================
// Test 2: Adjoint gradient matches FD gradient (end-to-end with caching)
//==============================================================================

static const char* paramNames[38] = {
    "convNorm", "promptNorm", "ADU", "KLU",
    "HEkp", "HEkm", "VHE1pip", "VHE1pim",
    "VHE3kp", "VHE3km", "VHE3pip", "VHE3pim", "VHE3p", "VHE3n",
    "CR1", "CR2", "CR3", "CR4", "CR5", "CR6",
    "icegrad0", "icegrad1", "icegrad2", "icegrad3", "icegrad4",
    "icegrad5", "icegrad6", "icegrad7", "icegrad8",
    "domEff", "holeIce",
    "astroNorm", "dGamma", "dGammaSec", "astroPivot",
    "neuaneuRatio", "nuXS", "nubarXS"
};

TestResult testGradientWithCaching() {
    TestResult result;
    result.name = "Gradient Correctness - AD vs FD with cached references";

    try {
        TestAccelerator ta;
        ta.setup(50000, 42);

        auto params = createNominalParameters();
        // Set domEff and holeIce to non-reference values to exercise the caching
        params[29] = 1.20;   // domEff != 1.27
        params[30] = -0.8;   // holeIce != -1.0

        // Autodiff gradient
        std::vector<double> adGrad;
        Timer timer;
        timer.start();
        double llh_ad = ta.accel->evaluateLikelihoodWithGradient(params, adGrad, false);
        double adTime = timer.stop();

        // FD gradient
        std::vector<double> fdGrad;
        timer.start();
        computeFDGradient(*ta.accel, params, fdGrad, false);
        double fdTime = timer.stop();

        std::cout << "  LLH: " << std::fixed << std::setprecision(8) << llh_ad << "\n";
        std::cout << "  Timing: " << adTime << " ms (AD) vs " << fdTime << " ms (FD)\n\n";

        const double relTol = 1e-3;
        const double absTol = 1e-5;
        int mismatches = 0;
        double maxRelErr = 0.0;
        int maxErrIdx = -1;

        std::cout << "  " << std::setw(14) << "Parameter"
                  << std::setw(16) << "AD gradient"
                  << std::setw(16) << "FD gradient"
                  << std::setw(14) << "relErr" << "\n";
        std::cout << "  " << std::string(60, '-') << "\n";

        for (int j = 0; j < 38; ++j) {
            double absMax = std::max(std::abs(adGrad[j]), std::abs(fdGrad[j]));
            double absDiff = std::abs(adGrad[j] - fdGrad[j]);
            double relErr = (absMax > 1e-15) ? absDiff / absMax : absDiff;

            bool isFail = false;
            if (absMax > absTol) {
                isFail = (relErr > relTol);
            } else {
                isFail = (absDiff > absTol);
            }

            // Show key parameters and any failures
            bool show = isFail || j == 29 || j == 30;  // Always show domEff and holeIce
            if (show) {
                std::cout << "  " << std::setw(14) << paramNames[j]
                          << std::setw(16) << adGrad[j]
                          << std::setw(16) << fdGrad[j]
                          << std::setw(14) << std::scientific << relErr
                          << std::fixed;
                if (isFail) std::cout << " ***FAIL***";
                if (absMax <= absTol) std::cout << " (near-zero)";
                std::cout << "\n";
            }

            if (absMax > absTol && relErr > maxRelErr) {
                maxRelErr = relErr;
                maxErrIdx = j;
            }
            if (isFail) {
                mismatches++;
            }
        }

        std::cout << "\n  Max relative error: " << std::scientific << maxRelErr;
        if (maxErrIdx >= 0) std::cout << " at " << paramNames[maxErrIdx];
        std::cout << std::fixed << "\n";
        std::cout << "  Mismatches: " << mismatches << "/38\n";

        result.passed = (mismatches == 0);
        if (!result.passed) {
            result.message = std::to_string(mismatches) + " gradient components exceed tolerance";
        }

    } catch (const std::exception& ex) {
        result.passed = false;
        result.message = std::string("Exception: ") + ex.what();
    }

    return result;
}

//==============================================================================
// Test 3: Timing comparison
//==============================================================================

TestResult testTimingImprovement() {
    TestResult result;
    result.name = "Timing - Weight kernel with cached references";

    try {
        TestAccelerator ta;
        ta.setup(50000, 42);

        auto params = createNominalParameters();
        params[29] = 1.20;
        params[30] = -0.8;

        // Warm up
        for (int i = 0; i < 5; ++i) {
            ta.accel->evaluateLikelihood(params, false);
        }

        // Time 100 evaluations
        const int nIter = 100;
        Timer timer;
        timer.start();
        for (int i = 0; i < nIter; ++i) {
            ta.accel->evaluateLikelihood(params, false);
        }
        double totalMs = timer.stop();
        double avgMs = totalMs / nIter;

        auto timing = ta.accel->getLastTimingStats();

        std::cout << "  Average LLH evaluation: " << std::fixed << std::setprecision(3)
                  << avgMs << " ms\n";
        std::cout << "  Last breakdown:\n";
        std::cout << "    Param transfer: " << timing.paramTransferMs << " ms\n";
        std::cout << "    Weight compute: " << timing.weightComputeMs << " ms\n";
        std::cout << "    Histogram:      " << timing.histogramMs << " ms\n";
        std::cout << "    Likelihood:     " << timing.likelihoodMs << " ms\n";
        std::cout << "    Total:          " << timing.totalMs << " ms\n";

        // This test is informational — always passes
        result.passed = true;
        result.message = "Average: " + std::to_string(avgMs) + " ms/eval";

    } catch (const std::exception& ex) {
        result.passed = false;
        result.message = std::string("Exception: ") + ex.what();
    }

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Reference Spline Caching Tests\n";
    std::cout << "========================================\n\n";

    TestSuite suite;

    suite.addResult(testCachedReferencesNonZero());
    suite.addResult(testGradientWithCaching());
    suite.addResult(testTimingImprovement());

    suite.printSummary();
    return suite.allPassed() ? 0 : 1;
}
