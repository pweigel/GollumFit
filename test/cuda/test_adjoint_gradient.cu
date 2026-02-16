/**
 * @file test_adjoint_gradient.cu
 * @brief Validate adjoint-method GPU gradient against finite differences.
 *
 * Tests:
 * 1. Gradient at nominal parameters (no prior) vs central FD
 * 2. Gradient at nominal parameters (with prior) vs central FD
 * 3. Gradient at perturbed parameters vs central FD
 * 4. Timing comparison: adjoint vs finite-difference gradient
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include "cuda/GPUFitAccelerator.h"
#include "Event.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace gollumfit::gpu;

//==============================================================================
// Mock Event Generator (same as test_integration.cu)
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
// Helper: compute FD gradient using the accelerator
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
// Helper: set up a fully-configured accelerator
//==============================================================================

struct TestAccelerator {
    std::unique_ptr<GPUFitAccelerator> accel;
    HistogramConfig histConfig;

    void setup(int numEvents, int seed, bool withPriors) {
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

        if (withPriors) {
            std::vector<PriorConfig> priors(38);
            for (int i = 0; i < 38; ++i) {
                priors[i].hasPrior = true;
                priors[i].mean = 0.0;
                priors[i].sigma = 1.0;
            }
            // Disable priors on normalization parameters
            priors[0].hasPrior = false;   // convNorm
            priors[1].hasPrior = false;   // promptNorm
            priors[31].hasPrior = false;  // astroNorm
            priors[35].hasPrior = false;  // neuaneu ratio
            accel->setPriors(priors);
        }
    }
};

//==============================================================================
// Parameter names for readable output
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

//==============================================================================
// Test 1: Gradient at nominal (no prior)
//==============================================================================

TestResult testGradientNominalNoPrior() {
    TestResult result;
    result.name = "Adjoint Gradient - Nominal (no prior)";

    try {
        TestAccelerator ta;
        ta.setup(50000, 42, false);

        auto params = createNominalParameters();

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

        // Also verify LLH matches
        double llh_fwd = ta.accel->evaluateLikelihood(params, false);

        result.passed = true;

        // Check LLH agreement
        double llhRelErr = std::abs(llh_ad - llh_fwd) / std::max(std::abs(llh_ad), 1e-15);
        if (llhRelErr > 1e-10) {
            std::cout << "  WARNING: LLH mismatch: AD=" << llh_ad << " FWD=" << llh_fwd
                      << " relErr=" << llhRelErr << "\n";
        }

        // Check gradient agreement
        // Use combined absolute + relative tolerance: skip relative check when
        // both values are below absTol (FD noise floor ~ eps*f/eps ~ machine_eps*f)
        const double relTol = 1e-3;
        const double absTol = 1e-5;  // below this, both AD and FD are in the noise
        int mismatches = 0;
        double maxRelErr = 0.0;
        int maxErrIdx = -1;

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  LLH: " << llh_ad << " (AD), " << llh_fwd << " (forward)\n";
        std::cout << "  Timing: " << adTime << " ms (AD) vs " << fdTime << " ms (FD) = "
                  << fdTime / adTime << "x speedup\n\n";

        std::cout << "  " << std::setw(14) << "Parameter"
                  << std::setw(16) << "AD gradient"
                  << std::setw(16) << "FD gradient"
                  << std::setw(14) << "relErr" << "\n";
        std::cout << "  " << std::string(60, '-') << "\n";

        for (int j = 0; j < 38; ++j) {
            double absMax = std::max(std::abs(adGrad[j]), std::abs(fdGrad[j]));
            double absDiff = std::abs(adGrad[j] - fdGrad[j]);
            double relErr = (absMax > 1e-15) ? absDiff / absMax : absDiff;

            // Skip relative error check if both values are below the absolute noise floor
            bool isFail = false;
            if (absMax > absTol) {
                isFail = (relErr > relTol);
            } else {
                // Both essentially zero — check absolute difference only
                isFail = (absDiff > absTol);
            }

            bool show = isFail || (relErr > 1e-4 && absMax > absTol);
            if (show || j < 5 || j == 31 || j == 35) {
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

        std::cout << "\n  Max relative error (significant): " << std::scientific << maxRelErr
                  << (maxErrIdx >= 0 ? std::string(" at param ") + std::to_string(maxErrIdx) + " (" + paramNames[maxErrIdx] + ")" : " (none)")
                  << "\n" << std::fixed;
        std::cout << "  Mismatches (>" << relTol << "): " << mismatches << "/38\n";

        if (mismatches > 0) {
            result.passed = false;
            result.message = std::to_string(mismatches) + " gradient components exceed tolerance";
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }

    return result;
}

//==============================================================================
// Test 2: Gradient at nominal (with prior)
//==============================================================================

TestResult testGradientNominalWithPrior() {
    TestResult result;
    result.name = "Adjoint Gradient - Nominal (with prior)";

    try {
        TestAccelerator ta;
        ta.setup(50000, 42, true);

        auto params = createNominalParameters();

        std::vector<double> adGrad;
        double llh_ad = ta.accel->evaluateLikelihoodWithGradient(params, adGrad, true);

        std::vector<double> fdGrad;
        computeFDGradient(*ta.accel, params, fdGrad, true);

        result.passed = true;
        const double relTol = 1e-3;
        const double absTol = 1e-5;
        int mismatches = 0;
        double maxRelErr = 0.0;
        int maxErrIdx = -1;

        std::cout << "  LLH with prior: " << llh_ad << "\n\n";

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

            if (absMax > absTol && relErr > maxRelErr) {
                maxRelErr = relErr;
                maxErrIdx = j;
            }
            if (isFail) {
                mismatches++;
                std::cout << "  FAIL: " << paramNames[j]
                          << " AD=" << adGrad[j] << " FD=" << fdGrad[j]
                          << " relErr=" << std::scientific << relErr << std::fixed << "\n";
            }
        }

        std::cout << "  Max relative error (significant): " << std::scientific << maxRelErr
                  << (maxErrIdx >= 0 ? std::string(" at ") + paramNames[maxErrIdx] : " (none)")
                  << std::fixed << "\n";
        std::cout << "  Mismatches: " << mismatches << "/38\n";

        if (mismatches > 0) {
            result.passed = false;
            result.message = std::to_string(mismatches) + " gradient components exceed tolerance";
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }

    return result;
}

//==============================================================================
// Test 3: Gradient at perturbed parameters
//==============================================================================

TestResult testGradientPerturbed() {
    TestResult result;
    result.name = "Adjoint Gradient - Perturbed parameters";

    try {
        TestAccelerator ta;
        ta.setup(50000, 42, true);

        // Perturb parameters away from nominal
        auto params = createNominalParameters();
        TestRNG rng(999);
        params[0] = 1.0 + rng.uniform(-0.1, 0.1);   // convNorm
        params[1] = 1.0 + rng.uniform(-0.1, 0.1);   // promptNorm
        params[2] = rng.uniform(-0.3, 0.3);          // ADU
        params[3] = rng.uniform(-0.3, 0.3);          // KLU
        for (int i = 4; i <= 13; ++i)
            params[i] = rng.uniform(-0.2, 0.2);      // hadronic
        for (int i = 14; i <= 19; ++i)
            params[i] = rng.uniform(-0.2, 0.2);      // CR
        for (int i = 20; i <= 28; ++i)
            params[i] = rng.uniform(-0.3, 0.3);      // ice grad
        params[31] = 1.0 + rng.uniform(-0.2, 0.2);  // astroNorm
        params[32] = rng.uniform(-0.1, 0.1);         // dGamma
        params[33] = rng.uniform(-0.1, 0.1);         // dGammaSec
        params[34] = rng.uniform(-0.1, 0.1);         // astroPivot
        params[35] = 1.0 + rng.uniform(-0.2, 0.2);  // neuaneu

        std::vector<double> adGrad;
        double llh_ad = ta.accel->evaluateLikelihoodWithGradient(params, adGrad, true);

        std::vector<double> fdGrad;
        computeFDGradient(*ta.accel, params, fdGrad, true);

        result.passed = true;
        const double relTol = 1e-3;
        const double absTol = 1e-5;
        int mismatches = 0;
        double maxRelErr = 0.0;
        int maxErrIdx = -1;

        std::cout << "  LLH at perturbed point: " << llh_ad << "\n";

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

            if (absMax > absTol && relErr > maxRelErr) {
                maxRelErr = relErr;
                maxErrIdx = j;
            }
            if (isFail) {
                mismatches++;
                std::cout << "  FAIL: " << paramNames[j]
                          << " AD=" << adGrad[j] << " FD=" << fdGrad[j]
                          << " relErr=" << std::scientific << relErr << std::fixed << "\n";
            }
        }

        std::cout << "  Max relative error (significant): " << std::scientific << maxRelErr
                  << (maxErrIdx >= 0 ? std::string(" at ") + paramNames[maxErrIdx] : " (none)")
                  << std::fixed << "\n";
        std::cout << "  Mismatches: " << mismatches << "/38\n";

        if (mismatches > 0) {
            result.passed = false;
            result.message = std::to_string(mismatches) + " gradient components exceed tolerance";
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }

    return result;
}

//==============================================================================
// Test 4: Timing comparison
//==============================================================================

TestResult testGradientTiming() {
    TestResult result;
    result.name = "Adjoint Gradient - Timing";

    try {
        TestAccelerator ta;
        ta.setup(50000, 42, false);

        auto params = createNominalParameters();
        std::vector<double> grad;

        // Warmup
        ta.accel->evaluateLikelihoodWithGradient(params, grad, false);
        ta.accel->evaluateLikelihood(params, false);

        // Time adjoint gradient (multiple iterations)
        const int nIter = 5;
        Timer timer;

        timer.start();
        for (int i = 0; i < nIter; ++i) {
            ta.accel->evaluateLikelihoodWithGradient(params, grad, false);
        }
        double adTotalMs = timer.stop();
        double adPerIter = adTotalMs / nIter;

        // Time FD gradient (1 iteration = 77 evaluations)
        timer.start();
        std::vector<double> fdGrad;
        computeFDGradient(*ta.accel, params, fdGrad, false);
        double fdTotalMs = timer.stop();

        // Time single LLH evaluation
        timer.start();
        for (int i = 0; i < nIter; ++i) {
            ta.accel->evaluateLikelihood(params, false);
        }
        double fwdTotalMs = timer.stop();
        double fwdPerIter = fwdTotalMs / nIter;

        double speedup = fdTotalMs / adPerIter;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Single LLH evaluation: " << fwdPerIter << " ms\n";
        std::cout << "  Adjoint gradient:      " << adPerIter << " ms ("
                  << adPerIter / fwdPerIter << "x single eval)\n";
        std::cout << "  FD gradient (77 evals): " << fdTotalMs << " ms\n";
        std::cout << "  Speedup (FD/adjoint):  " << speedup << "x\n";

        result.passed = true;  // Timing test always passes, just reports numbers
        result.gpuTime = adPerIter;
        result.cpuTime = fdTotalMs;

    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Adjoint Method Gradient Validation Tests\n";
    std::cout << "========================================\n\n";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n\n";

    TestSuite suite;

    suite.addResult(testGradientNominalNoPrior());
    suite.addResult(testGradientNominalWithPrior());
    suite.addResult(testGradientPerturbed());
    suite.addResult(testGradientTiming());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
