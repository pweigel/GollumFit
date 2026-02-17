/**
 * @file GPUFitAccelerator.cu
 * @brief Implementation of the main GPU accelerator for GollumFit.
 */

#include "cuda/GPUFitAccelerator.h"

#ifdef GOLLUMFIT_USE_CUDA

#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace gollumfit {
namespace gpu {

//==============================================================================
// External Kernel Declarations
//==============================================================================

// From event_weighting.cu
extern void launchEventWeightingKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    double* d_weights,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
);

extern void launchEventWeightingWithSquaresKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    double* d_weights,
    double* d_weightsSquared,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
);

// From histogram_reduction.cu
extern void launchHistogramAccumulationWithSquares(
    const double* d_weights,
    const double* d_weightsSquared,
    const int32_t* d_binIndices,
    double* d_binSums,
    double* d_binSqSums,
    int numEvents,
    int numBins,
    cudaStream_t stream
);

extern void launchHistogramAccumulationShared(
    const double* d_weights,
    const int32_t* d_binIndices,
    double* d_binSums,
    int numEvents,
    int numBins,
    cudaStream_t stream
);

extern void downloadHistogram(
    const double* d_histogram,
    double* h_histogram,
    int numBins,
    cudaStream_t stream
);

extern double getTotalHistogramSum(
    const double* d_histogram,
    int numBins,
    cudaStream_t stream
);

// From likelihood.cu
extern double computeTotalSAYLikelihood(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    int numBins,
    cudaStream_t stream
);

extern double computeTotalPrior(
    const double* d_params,
    const double* d_priorMeans,
    const double* d_priorSigmas,
    const int* d_priorFlags,
    int numParams,
    cudaStream_t stream
);

// From bin_sensitivity.cu
extern void launchBinSensitivities(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    double* d_adjoint_wsum,
    double* d_adjoint_w2sum,
    int numBins,
    cudaStream_t stream
);

// From event_weighting_gradient.cu
// Number of intermediate fields per event for gradient computation
constexpr int GRAD_NUM_INTERMEDIATE_FIELDS = 26;

extern void launchEventWeightGradientKernel(
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
);

// From precompute_reference_splines.cu
extern void launchPrecomputeReferenceSplines(
    GPUEventDataSoA& events,
    const GPUSplineLookup& splines,
    int numEvents,
    cudaStream_t stream
);

//==============================================================================
// GPUFitAccelerator Implementation
//==============================================================================

GPUFitAccelerator::GPUFitAccelerator(const GPUAcceleratorConfig& config)
    : config_(config)
{
    // Set device
    CUDA_CHECK(cudaSetDevice(config_.deviceId));

    // Create streams
    streams_.reserve(config_.numStreams);
    for (int i = 0; i < config_.numStreams; ++i) {
        streams_.emplace_back();
    }

    // Allocate pinned memory for parameter transfers
    h_params_ = PinnedPtr<double>(NUM_FIT_PARAMS);

    // Allocate device memory for parameters
    d_params_ = DevicePtr<double>(NUM_FIT_PARAMS);

    // Initialize spline lookup to empty
    splineLookup_.clear();
}

GPUFitAccelerator::~GPUFitAccelerator() {
    // RAII handles cleanup via DevicePtr/PinnedPtr destructors
    // Explicit synchronization to ensure all operations complete
    if (initialized_) {
        cudaDeviceSynchronize();
    }
}

GPUFitAccelerator::GPUFitAccelerator(GPUFitAccelerator&& other) noexcept
    : config_(std::move(other.config_)),
      histConfig_(std::move(other.histConfig_)),
      initialized_(other.initialized_),
      eventManager_(std::move(other.eventManager_)),
      splineManager_(std::move(other.splineManager_)),
      splineNameToIndex_(std::move(other.splineNameToIndex_)),
      splineLookup_(other.splineLookup_),
      d_binSums_(std::move(other.d_binSums_)),
      d_binSqSums_(std::move(other.d_binSqSums_)),
      d_dataCount_(std::move(other.d_dataCount_)),
      d_weights_(std::move(other.d_weights_)),
      d_weightsSquared_(std::move(other.d_weightsSquared_)),
      d_params_(std::move(other.d_params_)),
      d_priorMeans_(std::move(other.d_priorMeans_)),
      d_priorSigmas_(std::move(other.d_priorSigmas_)),
      d_priorFlags_(std::move(other.d_priorFlags_)),
      priorsConfigured_(other.priorsConfigured_),
      h_params_(std::move(other.h_params_)),
      d_adjoint_wsum_(std::move(other.d_adjoint_wsum_)),
      d_adjoint_w2sum_(std::move(other.d_adjoint_w2sum_)),
      d_gradient_(std::move(other.d_gradient_)),
      h_gradient_(std::move(other.h_gradient_)),
      h_priorMeans_(std::move(other.h_priorMeans_)),
      h_priorSigmas_(std::move(other.h_priorSigmas_)),
      h_priorFlags_(std::move(other.h_priorFlags_)),
      streams_(std::move(other.streams_)),
      lastTiming_(other.lastTiming_)
{
    other.initialized_ = false;
    other.priorsConfigured_ = false;
    other.splineLookup_.clear();
}

GPUFitAccelerator& GPUFitAccelerator::operator=(GPUFitAccelerator&& other) noexcept {
    if (this != &other) {
        // Clean up current state
        if (initialized_) {
            cudaDeviceSynchronize();
        }

        config_ = std::move(other.config_);
        histConfig_ = std::move(other.histConfig_);
        initialized_ = other.initialized_;
        eventManager_ = std::move(other.eventManager_);
        splineManager_ = std::move(other.splineManager_);
        splineNameToIndex_ = std::move(other.splineNameToIndex_);
        splineLookup_ = other.splineLookup_;
        d_binSums_ = std::move(other.d_binSums_);
        d_binSqSums_ = std::move(other.d_binSqSums_);
        d_dataCount_ = std::move(other.d_dataCount_);
        d_weights_ = std::move(other.d_weights_);
        d_weightsSquared_ = std::move(other.d_weightsSquared_);
        d_params_ = std::move(other.d_params_);
        d_priorMeans_ = std::move(other.d_priorMeans_);
        d_priorSigmas_ = std::move(other.d_priorSigmas_);
        d_priorFlags_ = std::move(other.d_priorFlags_);
        priorsConfigured_ = other.priorsConfigured_;
        h_params_ = std::move(other.h_params_);
        d_adjoint_wsum_ = std::move(other.d_adjoint_wsum_);
        d_adjoint_w2sum_ = std::move(other.d_adjoint_w2sum_);
        d_gradient_ = std::move(other.d_gradient_);
        h_gradient_ = std::move(other.h_gradient_);
        h_priorMeans_ = std::move(other.h_priorMeans_);
        h_priorSigmas_ = std::move(other.h_priorSigmas_);
        h_priorFlags_ = std::move(other.h_priorFlags_);
        streams_ = std::move(other.streams_);
        lastTiming_ = other.lastTiming_;

        other.initialized_ = false;
        other.priorsConfigured_ = false;
        other.splineLookup_.clear();
    }
    return *this;
}

void GPUFitAccelerator::initialize(
    GPUEventDataManager&& eventManager,
    const HistogramConfig& histConfig
) {
    histConfig_ = histConfig;
    eventManager_ = std::make_unique<GPUEventDataManager>(std::move(eventManager));
    allocateDeviceMemory();
    initialized_ = true;
}

void GPUFitAccelerator::allocateDeviceMemory() {
    if (!eventManager_) {
        throw std::runtime_error("GPUFitAccelerator: event manager not set");
    }

    const int totalBins = histConfig_.totalBins();
    const size_t numEvents = eventManager_->getNumEvents();

    if (totalBins == 0) {
        throw std::runtime_error("GPUFitAccelerator: histogram has zero bins");
    }
    if (numEvents == 0) {
        throw std::runtime_error("GPUFitAccelerator: no events uploaded");
    }

    // Allocate histogram buffers
    d_binSums_ = DevicePtr<double>(totalBins);
    d_binSqSums_ = DevicePtr<double>(totalBins);
    d_dataCount_ = DevicePtr<double>(totalBins);

    // Allocate weight buffers
    d_weights_ = DevicePtr<double>(numEvents);
    d_weightsSquared_ = DevicePtr<double>(numEvents);

    // Allocate adjoint method buffers
    d_adjoint_wsum_ = DevicePtr<double>(totalBins);
    d_adjoint_w2sum_ = DevicePtr<double>(totalBins);
    d_gradient_ = DevicePtr<double>(NUM_FIT_PARAMS);
    h_gradient_ = PinnedPtr<double>(NUM_FIT_PARAMS);

    // Allocate gradient intermediate buffer (SoA: 26 doubles per event)
    d_gradIntermediate_ = DevicePtr<double>(GRAD_NUM_INTERMEDIATE_FIELDS * numEvents);

    // Initialize histograms to zero
    CUDA_CHECK(cudaMemset(d_binSums_.get(), 0, totalBins * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_binSqSums_.get(), 0, totalBins * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dataCount_.get(), 0, totalBins * sizeof(double)));
}

void GPUFitAccelerator::uploadDataHistogram(const std::vector<double>& dataCount) {
    const int totalBins = histConfig_.totalBins();

    if (static_cast<int>(dataCount.size()) != totalBins) {
        throw std::runtime_error("GPUFitAccelerator: data histogram size mismatch");
    }

    CUDA_CHECK(cudaMemcpy(
        d_dataCount_.get(),
        dataCount.data(),
        totalBins * sizeof(double),
        cudaMemcpyHostToDevice
    ));
}

void GPUFitAccelerator::setPriors(const std::vector<PriorConfig>& priors) {
    if (priors.size() != NUM_FIT_PARAMS) {
        throw std::runtime_error("GPUFitAccelerator: prior config size mismatch");
    }

    // Allocate device memory for priors
    d_priorMeans_ = DevicePtr<double>(NUM_FIT_PARAMS);
    d_priorSigmas_ = DevicePtr<double>(NUM_FIT_PARAMS);
    d_priorFlags_ = DevicePtr<int>(NUM_FIT_PARAMS);

    // Prepare host arrays
    std::vector<double> means(NUM_FIT_PARAMS);
    std::vector<double> sigmas(NUM_FIT_PARAMS);
    std::vector<int> flags(NUM_FIT_PARAMS);

    for (size_t i = 0; i < NUM_FIT_PARAMS; ++i) {
        means[i] = priors[i].mean;
        sigmas[i] = priors[i].sigma;
        flags[i] = priors[i].hasPrior ? 1 : 0;
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_priorMeans_.get(), means.data(),
                          NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_priorSigmas_.get(), sigmas.data(),
                          NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_priorFlags_.get(), flags.data(),
                          NUM_FIT_PARAMS * sizeof(int), cudaMemcpyHostToDevice));

    // Cache host copies for CPU-side prior gradient computation
    h_priorMeans_ = means;
    h_priorSigmas_ = sigmas;
    h_priorFlags_ = flags;

    priorsConfigured_ = true;
}

void GPUFitAccelerator::addNDPrior(const NDPriorConfig& config) {
    ndPriors_.push_back(config);
}

//==============================================================================
// Spline Management
//==============================================================================

int GPUFitAccelerator::uploadSpline(
    const std::string& name,
    int ndim,
    const int* orders,
    const std::vector<std::vector<double>>& knots,
    const std::vector<double>& coeffs,
    const int* nknots
) {
    int index = splineManager_.uploadSpline(ndim, orders, knots, coeffs, nknots);
    splineNameToIndex_[name] = index;
    return index;
}

const GPUSplineTable* GPUFitAccelerator::getSpline(const std::string& name) const {
    auto it = splineNameToIndex_.find(name);
    if (it == splineNameToIndex_.end()) {
        return nullptr;
    }
    return splineManager_.getDeviceSpline(it->second);
}

const GPUSplineTable* GPUFitAccelerator::getSpline(int index) const {
    return splineManager_.getDeviceSpline(index);
}

void GPUFitAccelerator::buildSplineLookup() {
    splineLookup_.clear();

    if (splineManager_.getNumSplines() == 0) {
        return;
    }

    // Map flux component names to GPU enum values
    struct FluxMapping {
        const char* name;
        GPUFluxComponent component;
    };
    FluxMapping fluxMappings[] = {
        {"atmConv", GPU_FLUX_CONV},
        {"atmPrompt", GPU_FLUX_PROMPT},
        {"diffuseAstro", GPU_FLUX_ASTRO}
    };

    // Map topology names to GPU enum values
    // Note: Topology::shower maps to string "shower" (index 0 = cascade/shower)
    struct TopoMapping {
        const char* name;
        GPUTopology topology;
    };
    TopoMapping topoMappings[] = {
        {"shower", GPU_TOPO_CASCADE},
        {"track", GPU_TOPO_TRACK}
    };

    // Build DOM efficiency spline lookup
    for (const auto& flux : fluxMappings) {
        for (const auto& topo : topoMappings) {
            std::string name = std::string("domeff_") + flux.name + "_" + topo.name;
            auto it = splineNameToIndex_.find(name);
            if (it != splineNameToIndex_.end()) {
                splineLookup_.domEffSplines[flux.component][topo.topology] =
                    splineManager_.getDeviceSpline(it->second);
                splineLookup_.hasSplines = true;
            }
        }
    }

    // Build hole ice spline lookup
    for (const auto& flux : fluxMappings) {
        for (const auto& topo : topoMappings) {
            std::string name = std::string("holeice_") + flux.name + "_" + topo.name;
            auto it = splineNameToIndex_.find(name);
            if (it != splineNameToIndex_.end()) {
                splineLookup_.holeIceSplines[flux.component][topo.topology] =
                    splineManager_.getDeviceSpline(it->second);
                splineLookup_.hasSplines = true;
            }
        }
    }

    // Build attenuation spline lookup
    // Attenuation splines are indexed by (FluxComponent, ParticleType)
    struct ParticleMapping {
        const char* name;
        GPUParticleType particleType;
    };
    ParticleMapping particleMappings[] = {
        {"numu", GPU_PTYPE_NUMU},
        {"numubar", GPU_PTYPE_NUMUBAR},
        {"nutau", GPU_PTYPE_NUTAU},
        {"nutaubar", GPU_PTYPE_NUTAUBAR}
    };

    for (const auto& flux : fluxMappings) {
        for (const auto& part : particleMappings) {
            std::string name = std::string("attenuation_") + flux.name + "_" + part.name;
            auto it = splineNameToIndex_.find(name);
            if (it != splineNameToIndex_.end()) {
                splineLookup_.attenSplines[flux.component][part.particleType] =
                    splineManager_.getDeviceSpline(it->second);
                splineLookup_.hasSplines = true;
            }
        }
    }

    // Set reference values
    splineLookup_.domEffReference = 1.27;
    splineLookup_.holeIceReference = -1.0;

    // Enable basis caching for DOM eff / hole ice splines
    // (all share the same knot vectors for dims 0 & 1)
    splineLookup_.basisCacheValid = splineLookup_.hasSplines;
    basisCacheValid_ = splineLookup_.basisCacheValid;

    if (splineLookup_.hasSplines) {
        std::cout << "  GPU spline lookup built successfully" << std::endl;
    }
}

void GPUFitAccelerator::precomputeReferenceSplines() {
    if (!initialized_ || !eventManager_) return;

    GPUEventDataSoA& events = eventManager_->getDeviceData();
    int numEvents = static_cast<int>(eventManager_->getNumEvents());

    cudaStream_t stream = streams_.empty() ? nullptr : streams_[0].get();

    launchPrecomputeReferenceSplines(events, splineLookup_, numEvents, stream);

    if (basisCacheValid_) {
        std::cout << "  Precomputed reference spline values for " << numEvents
                  << " events (basis cache enabled)" << std::endl;
    } else {
        std::cout << "  Precomputed reference spline values for " << numEvents
                  << " events" << std::endl;
    }
}

//==============================================================================
// Likelihood Evaluation
//==============================================================================

double GPUFitAccelerator::evaluateLikelihood(
    const std::vector<double>& params,
    bool includePrior
) {
    if (!initialized_) {
        throw std::runtime_error("GPUFitAccelerator: not initialized");
    }

    if (params.size() != NUM_FIT_PARAMS) {
        throw std::runtime_error("GPUFitAccelerator: parameter size mismatch");
    }

    cudaStream_t stream = streams_[0].get();

    // Reset timing
    lastTiming_ = TimingStats();
    cudaEvent_t startEvent, endEvent;
    float elapsedMs = 0.0f;  // cudaEventElapsedTime expects float*
    if (config_.enableProfiling) {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&endEvent);
    }

    // 1. Transfer parameters to GPU (async)
    if (config_.enableProfiling) cudaEventRecord(startEvent, stream);

    std::memcpy(h_params_.get(), params.data(), NUM_FIT_PARAMS * sizeof(double));
    CUDA_CHECK(cudaMemcpyAsync(
        d_params_.get(),
        h_params_.get(),
        NUM_FIT_PARAMS * sizeof(double),
        cudaMemcpyHostToDevice,
        stream
    ));

    if (config_.enableProfiling) {
        cudaEventRecord(endEvent, stream);
        cudaEventSynchronize(endEvent);
        cudaEventElapsedTime(&elapsedMs, startEvent, endEvent);
        lastTiming_.paramTransferMs = static_cast<double>(elapsedMs);
    }

    // 2. Compute event weights
    if (config_.enableProfiling) cudaEventRecord(startEvent, stream);
    computeWeights(stream);
    if (config_.enableProfiling) {
        cudaEventRecord(endEvent, stream);
        cudaEventSynchronize(endEvent);
        cudaEventElapsedTime(&elapsedMs, startEvent, endEvent);
        lastTiming_.weightComputeMs = static_cast<double>(elapsedMs);
    }

    // 3. Accumulate histogram
    if (config_.enableProfiling) cudaEventRecord(startEvent, stream);
    accumulateHistogram(stream);
    if (config_.enableProfiling) {
        cudaEventRecord(endEvent, stream);
        cudaEventSynchronize(endEvent);
        cudaEventElapsedTime(&elapsedMs, startEvent, endEvent);
        lastTiming_.histogramMs = static_cast<double>(elapsedMs);
    }

    // 4. Compute likelihood
    if (config_.enableProfiling) cudaEventRecord(startEvent, stream);
    double llh = computeLikelihood(includePrior, stream);
    if (config_.enableProfiling) {
        cudaEventRecord(endEvent, stream);
        cudaEventSynchronize(endEvent);
        cudaEventElapsedTime(&elapsedMs, startEvent, endEvent);
        lastTiming_.likelihoodMs = static_cast<double>(elapsedMs);

        lastTiming_.totalMs = lastTiming_.paramTransferMs +
                              lastTiming_.weightComputeMs +
                              lastTiming_.histogramMs +
                              lastTiming_.likelihoodMs;

        cudaEventDestroy(startEvent);
        cudaEventDestroy(endEvent);
    }

    return llh;
}

double GPUFitAccelerator::evaluateLikelihoodWithGradient(
    const std::vector<double>& params,
    std::vector<double>& gradient,
    bool includePrior
) {
    if (!initialized_) {
        throw std::runtime_error("GPUFitAccelerator: not initialized");
    }

    if (params.size() != NUM_FIT_PARAMS) {
        throw std::runtime_error("GPUFitAccelerator: parameter size mismatch");
    }

    gradient.resize(NUM_FIT_PARAMS);
    cudaStream_t stream = streams_[0].get();

    // 1. Transfer parameters to GPU
    std::memcpy(h_params_.get(), params.data(), NUM_FIT_PARAMS * sizeof(double));
    CUDA_CHECK(cudaMemcpyAsync(
        d_params_.get(), h_params_.get(),
        NUM_FIT_PARAMS * sizeof(double),
        cudaMemcpyHostToDevice, stream
    ));

    // 2. Forward pass: compute weights, histogram, likelihood (same as evaluateLikelihood)
    computeWeights(stream);
    accumulateHistogram(stream);
    double llh = computeLikelihood(includePrior, stream);

    // 3. Bin sensitivities: compute dSAY/d(w_sum) and dSAY/d(w2_sum) per bin
    const int totalBins = histConfig_.totalBins();
    launchBinSensitivities(
        d_dataCount_.get(),
        d_binSums_.get(),
        d_binSqSums_.get(),
        d_adjoint_wsum_.get(),
        d_adjoint_w2sum_.get(),
        totalBins,
        stream
    );

    // 4. Weight gradient kernel: compute d(-logL_data)/d(params) via adjoint contraction
    computeWeightsWithGradient(stream);

    // 5. Download gradient to host
    CUDA_CHECK(cudaMemcpyAsync(
        h_gradient_.get(), d_gradient_.get(),
        NUM_FIT_PARAMS * sizeof(double),
        cudaMemcpyDeviceToHost, stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy to output
    std::memcpy(gradient.data(), h_gradient_.get(), NUM_FIT_PARAMS * sizeof(double));

    // 6. Add prior gradient (CPU computation)
    if (includePrior) {
        addPriorGradient(gradient);
    }

    return llh;
}

void GPUFitAccelerator::addPriorGradient(std::vector<double>& gradient) {
    if (!priorsConfigured_) return;

    const double* p = h_params_.get();

    // Individual Gaussian priors: d(-logL_prior)/d(p_j) = (p_j - mu_j) / sigma_j^2
    // The prior log-likelihood is: -log(sigma) - 0.5*ln(2*pi) - 0.5*((x-mu)/sigma)^2
    // Its derivative w.r.t. x is: -(x-mu)/sigma^2
    // Since we minimize -logL, the gradient contribution is: +(x-mu)/sigma^2
    for (size_t j = 0; j < NUM_FIT_PARAMS; ++j) {
        if (h_priorFlags_[j] == 0) continue;
        double sigma = h_priorSigmas_[j];
        if (std::isinf(sigma) || std::isnan(sigma)) continue;
        double z = (p[j] - h_priorMeans_[j]) / sigma;
        gradient[j] += z / sigma;
    }

    // N-dimensional correlated Gaussian priors:
    // Prior = lnorm - 0.5 * z^T * invCorr * z  where z_i = (mean_i - x_i) / stddev_i
    // d(-prior)/d(x_k) = sum_i [ invCorr[row_k][i] * z_i / stddev_k ]
    // (since z_k = (mean_k - x_k)/stddev_k, dz_k/dx_k = -1/stddev_k)
    // More precisely: d(prior)/d(x_k) = sum_j [ z_j * invCorr[j][idx_k_in_group] / stddev_k
    //                                          + z_j * invCorr[idx_k_in_group][j] / stddev_k ] * (-0.5) * (-1/stddev_k)
    // Since invCorr is symmetric: d(prior)/d(x_k) = sum_j [ invCorr[idx_k][j] * z_j ] / stddev_k
    // And d(-prior)/d(x_k) = -d(prior)/d(x_k)
    for (const auto& nd : ndPriors_) {
        if (nd.size == 0) continue;

        // Compute z vector
        std::vector<double> divvy(nd.size);
        for (int i = 0; i < nd.size; i++) {
            divvy[i] = (nd.means[i] - p[nd.paramIndices[i]]) / nd.stddevs[i];
        }

        // For each parameter in the group
        for (int k = 0; k < nd.size; k++) {
            int paramIdx = nd.paramIndices[k];
            // d(prior)/d(x_k): the prior is -0.5 * z^T * C^-1 * z
            // d/dx_k of z_k = -1/stddev_k (z_k = (mean_k - x_k)/stddev_k)
            // d(prior)/d(x_k) = -0.5 * 2 * sum_j[C^-1_{k,j} * z_j] * (-1/stddev_k)
            //                 = sum_j[C^-1_{k,j} * z_j] / stddev_k
            double sum = 0.0;
            for (int j = 0; j < nd.size; j++) {
                sum += nd.inverseCorr[k * nd.size + j] * divvy[j];
            }
            // d(prior)/d(x_k) = sum / stddev_k
            // d(-prior)/d(x_k) = -sum / stddev_k  (we minimize -logL)
            gradient[paramIdx] += -sum / nd.stddevs[k];
        }
    }
}

//==============================================================================
// Internal Methods
//==============================================================================

void GPUFitAccelerator::computeWeights(cudaStream_t stream) {
    const GPUEventDataSoA& events = eventManager_->getDeviceData();
    const int numEvents = static_cast<int>(eventManager_->getNumEvents());

    // Check if totalNorm parameter is enabled (param index 37)
    bool enableTotalNorm = true;  // Always enable for now

    launchEventWeightingWithSquaresKernel(
        events,
        d_params_.get(),
        splineLookup_,
        d_weights_.get(),
        d_weightsSquared_.get(),
        numEvents,
        enableTotalNorm,
        stream
    );
}

void GPUFitAccelerator::computeWeightsWithGradient(cudaStream_t stream) {
    const GPUEventDataSoA& events = eventManager_->getDeviceData();
    const int numEvents = static_cast<int>(eventManager_->getNumEvents());
    bool enableTotalNorm = true;

    launchEventWeightGradientKernel(
        events,
        d_params_.get(),
        splineLookup_,
        d_adjoint_wsum_.get(),
        d_adjoint_w2sum_.get(),
        d_gradient_.get(),
        numEvents,
        enableTotalNorm,
        d_gradIntermediate_.get(),
        stream
    );
}

void GPUFitAccelerator::accumulateHistogram(cudaStream_t stream) {
    const GPUEventDataSoA& events = eventManager_->getDeviceData();
    const int numEvents = static_cast<int>(eventManager_->getNumEvents());
    const int totalBins = histConfig_.totalBins();

    if (config_.useSharedMemoryHistogram) {
        // Use shared memory optimization for small histograms
        launchHistogramAccumulationShared(
            d_weights_.get(),
            events.binIndex,
            d_binSums_.get(),
            numEvents,
            totalBins,
            stream
        );

        // Need separate kernel for squared weights
        launchHistogramAccumulationShared(
            d_weightsSquared_.get(),
            events.binIndex,
            d_binSqSums_.get(),
            numEvents,
            totalBins,
            stream
        );
    } else {
        // Use standard atomic accumulation
        launchHistogramAccumulationWithSquares(
            d_weights_.get(),
            d_weightsSquared_.get(),
            events.binIndex,
            d_binSums_.get(),
            d_binSqSums_.get(),
            numEvents,
            totalBins,
            stream
        );
    }
}

double GPUFitAccelerator::computeLikelihood(bool includePrior, cudaStream_t stream) {
    const int totalBins = histConfig_.totalBins();

    // Compute SAY likelihood
    double dataLLH = computeTotalSAYLikelihood(
        d_dataCount_.get(),
        d_binSums_.get(),
        d_binSqSums_.get(),
        totalBins,
        stream
    );

    // Compute prior if requested
    double priorLLH = 0.0;
    if (includePrior && priorsConfigured_) {
        // PhysTools base-case correction: both FixedSizePriorSet::priorEvaluator
        // and ArbitraryPriorSet::priorEvaluator have base cases that return 1.0
        // instead of 0.0. These accumulate additively in the log-likelihood sum,
        // contributing +2.0 to the CPU's prior total. We match this for consistency.
        priorLLH = 2.0;

        // Individual Gaussian priors (computed on GPU)
        double gaussPrior = computeTotalPrior(
            d_params_.get(),
            d_priorMeans_.get(),
            d_priorSigmas_.get(),
            d_priorFlags_.get(),
            NUM_FIT_PARAMS,
            stream
        );
        priorLLH += gaussPrior;

        // N-dimensional correlated Gaussian priors (computed on CPU)
        // h_params_ already has the current parameter values
        for (size_t k = 0; k < ndPriors_.size(); k++) {
            priorLLH += ndPriors_[k].evaluate(h_params_.get());
        }
    }

    // Return negative log-likelihood for minimization
    return -(dataLLH + priorLLH);
}

//==============================================================================
// Histogram Access
//==============================================================================

void GPUFitAccelerator::getExpectationHistogram(std::vector<double>& histogram) const {
    const int totalBins = histConfig_.totalBins();
    histogram.resize(totalBins);

    downloadHistogram(d_binSums_.get(), histogram.data(), totalBins, nullptr);
}

void GPUFitAccelerator::getWeightSquaredHistogram(std::vector<double>& histogram) const {
    const int totalBins = histConfig_.totalBins();
    histogram.resize(totalBins);

    downloadHistogram(d_binSqSums_.get(), histogram.data(), totalBins, nullptr);
}

void GPUFitAccelerator::getEventWeights(std::vector<double>& weights) const {
    size_t n = getNumEvents();
    weights.resize(n);

    CUDA_CHECK(cudaMemcpy(weights.data(), d_weights_.get(),
                          n * sizeof(double), cudaMemcpyDeviceToHost));
}

//==============================================================================
// Utility Functions
//==============================================================================

size_t GPUFitAccelerator::getNumEvents() const {
    return eventManager_ ? eventManager_->getNumEvents() : 0;
}

int GPUFitAccelerator::getNumBins() const {
    return histConfig_.totalBins();
}

GPUDeviceInfo GPUFitAccelerator::getDeviceInfo() const {
    return GPUDeviceInfo::query(config_.deviceId);
}

void GPUFitAccelerator::synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUFitAccelerator::reset() {
    const int totalBins = histConfig_.totalBins();

    if (d_binSums_.get() != nullptr) {
        CUDA_CHECK(cudaMemset(d_binSums_.get(), 0, totalBins * sizeof(double)));
    }
    if (d_binSqSums_.get() != nullptr) {
        CUDA_CHECK(cudaMemset(d_binSqSums_.get(), 0, totalBins * sizeof(double)));
    }
}

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA
