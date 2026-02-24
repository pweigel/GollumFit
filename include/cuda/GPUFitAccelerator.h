/**
 * @file GPUFitAccelerator.h
 * @brief Main GPU accelerator interface for GollumFit.
 *
 * This class provides the bridge between the CPU-based GollumFit framework
 * and the GPU-accelerated kernels. It manages:
 * - Event data transfers (SoA layout on GPU)
 * - Spline table management
 * - Likelihood and gradient evaluation
 * - Memory pools and CUDA streams
 */

#ifndef GOLLUMFIT_GPU_FIT_ACCELERATOR_H
#define GOLLUMFIT_GPU_FIT_ACCELERATOR_H

#ifdef GOLLUMFIT_USE_CUDA

#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include "cuda/GPUSplineTable.h"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>

namespace gollumfit {
namespace gpu {

//==============================================================================
// Configuration Structures
//==============================================================================

/**
 * @brief Configuration for GPU accelerator
 */
struct GPUAcceleratorConfig {
    int deviceId = 0;                    ///< GPU device ID to use
    bool useSharedMemoryHistogram = false; ///< Use shared memory for histogram (small bins)
    bool enableProfiling = false;        ///< Enable CUDA profiling
    bool useCudaGraphs = false;          ///< Use CUDA graphs for kernel sequences
    size_t maxEventsPerBatch = 0;        ///< Max events per batch (0 = all at once)
    int numStreams = 2;                  ///< Number of CUDA streams for pipelining
};

/**
 * @brief Histogram binning configuration
 */
struct HistogramConfig {
    int nBinsEnergy = 0;                 ///< Number of energy bins
    int nBinsZenith = 0;                 ///< Number of zenith bins
    int nBinsTopology = 0;               ///< Number of topology bins (usually 2)

    // Bin edges
    std::vector<double> energyEdges;     ///< Energy bin edges
    std::vector<double> zenithEdges;     ///< Zenith bin edges

    int totalBins() const {
        return nBinsEnergy * nBinsZenith * nBinsTopology;
    }
};

/**
 * @brief Prior configuration for a single parameter
 */
struct PriorConfig {
    bool hasPrior = false;
    double mean = 0.0;
    double sigma = 1.0;
};

/**
 * @brief N-dimensional correlated Gaussian prior
 *
 * Matches PhysTools GaussianNDPrior: lnorm - 0.5 * z^T * inv(corr) * z
 * where z[i] = (mean[i] - x[i]) / stddev[i].
 * Computed on CPU since the matrices are small.
 */
struct NDPriorConfig {
    std::vector<int> paramIndices;    ///< GPU parameter indices for this group
    std::vector<double> means;        ///< Prior means
    std::vector<double> stddevs;      ///< Prior standard deviations
    std::vector<double> inverseCorr;  ///< Flattened inverse correlation matrix (row-major)
    double lnorm = 0.0;              ///< Log normalization constant
    int size = 0;                     ///< Number of parameters in this group

    /// Evaluate the ND prior given parameter values
    double evaluate(const double* params) const {
        if (size == 0) return 0.0;
        std::vector<double> divvy(size);
        for (int i = 0; i < size; i++) {
            divvy[i] = (means[i] - params[paramIndices[i]]) / stddevs[i];
        }
        double quadForm = 0.0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                quadForm += divvy[i] * inverseCorr[i * size + j] * divvy[j];
            }
        }
        return lnorm - 0.5 * quadForm;
    }
};

//==============================================================================
// GPU Fit Accelerator Class
//==============================================================================

/**
 * @brief Main GPU acceleration interface for GollumFit
 *
 * This class encapsulates all GPU operations and provides a clean interface
 * for the CPU-based GollumFit to call into. It handles:
 *
 * 1. One-time setup:
 *    - Event data transfer (AoS → SoA transformation)
 *    - Spline table upload
 *    - Histogram configuration
 *
 * 2. Per-iteration operations:
 *    - Parameter transfer (38 doubles)
 *    - Weight computation
 *    - Histogram accumulation
 *    - Likelihood evaluation
 *    - Gradient computation (optional)
 *
 * Usage:
 * @code
 *   GPUFitAccelerator gpu(config);
 *   gpu.initialize(events, histConfig);
 *   gpu.uploadSpline("domeff", splineData);
 *
 *   // During fitting:
 *   double llh = gpu.evaluateLikelihood(params, dataHist, includePrior);
 *   // Or with gradients:
 *   double llh = gpu.evaluateLikelihoodWithGradient(params, dataHist, gradient);
 * @endcode
 */
class GPUFitAccelerator {
public:
    /**
     * @brief Construct accelerator with configuration
     */
    explicit GPUFitAccelerator(const GPUAcceleratorConfig& config = GPUAcceleratorConfig());

    /**
     * @brief Destructor - cleans up all GPU resources
     */
    ~GPUFitAccelerator();

    // Non-copyable
    GPUFitAccelerator(const GPUFitAccelerator&) = delete;
    GPUFitAccelerator& operator=(const GPUFitAccelerator&) = delete;

    // Movable
    GPUFitAccelerator(GPUFitAccelerator&& other) noexcept;
    GPUFitAccelerator& operator=(GPUFitAccelerator&& other) noexcept;

    //==========================================================================
    // Initialization
    //==========================================================================

    /**
     * @brief Initialize with event data and histogram configuration
     *
     * This transfers all event data to the GPU in SoA format.
     * Should be called once before fitting begins.
     *
     * @tparam EventContainer Container of Event objects (e.g., std::deque<Event>)
     * @param events MC event container
     * @param histConfig Histogram binning configuration
     */
    template<typename EventContainer>
    void initialize(const EventContainer& events, const HistogramConfig& histConfig);

    /**
     * @brief Initialize from pre-built GPU event data
     *
     * Use this if event data is already in GPU format.
     *
     * @param eventManager Pre-initialized event data manager
     * @param histConfig Histogram configuration
     */
    void initialize(GPUEventDataManager&& eventManager, const HistogramConfig& histConfig);

    /**
     * @brief Upload observed data histogram
     *
     * @param dataCount Observed counts per bin [totalBins]
     */
    void uploadDataHistogram(const std::vector<double>& dataCount);

    /**
     * @brief Set prior configuration for parameters
     *
     * @param priors Prior configuration for each parameter [NUM_FIT_PARAMS]
     */
    void setPriors(const std::vector<PriorConfig>& priors);

    /**
     * @brief Add an N-dimensional correlated Gaussian prior
     *
     * @param config ND prior configuration with inverse correlation matrix
     */
    void addNDPrior(const NDPriorConfig& config);

    //==========================================================================
    // Spline Management
    //==========================================================================

    /**
     * @brief Upload a spline table to GPU
     *
     * @param name Spline identifier (e.g., "domeff", "holeice")
     * @param ndim Number of dimensions
     * @param orders B-spline orders per dimension
     * @param knots Knot vectors per dimension
     * @param coeffs Coefficient array
     * @param nknots Number of knots per dimension
     * @return Spline index for later reference
     */
    int uploadSpline(
        const std::string& name,
        int ndim,
        const int* orders,
        const std::vector<std::vector<double>>& knots,
        const std::vector<double>& coeffs,
        const int* nknots
    );

    /**
     * @brief Get device pointer to uploaded spline
     */
    const GPUSplineTable* getSpline(const std::string& name) const;
    const GPUSplineTable* getSpline(int index) const;

    /**
     * @brief Build the spline lookup table for kernel access
     *
     * Must be called after all splines are uploaded. This builds a lookup
     * structure that maps (flux component, topology) to spline pointers,
     * which is passed to the event weighting kernel.
     */
    void buildSplineLookup();

    /**
     * @brief Precompute reference spline evaluations for all events.
     *
     * Evaluates DOM efficiency and hole ice splines at their fixed reference
     * values (1.27 and -1.0 respectively) for each event's (log10Energy, cosZenith).
     * Results are cached in the GPUEventDataSoA fields (cachedDOMEff*, cachedHoleIce*)
     * so that the weight and gradient kernels can skip these 6 evaluations per event.
     *
     * Must be called after buildSplineLookup(). Only needs to be called once.
     */
    void precomputeReferenceSplines();

    /**
     * @brief Override the basis cache valid flag (for benchmarking).
     *
     * When set to false, the weight kernels use the full spline evaluation
     * fallback path instead of the cached basis path.
     */
    void setBasisCacheValid(bool valid) {
        splineLookup_.basisCacheValid = valid;
        basisCacheValid_ = valid;
    }

    //==========================================================================
    // Likelihood Evaluation
    //==========================================================================

    /**
     * @brief Compute negative log-likelihood
     *
     * This is the main entry point for likelihood evaluation. It:
     * 1. Transfers parameters to GPU
     * 2. Computes event weights
     * 3. Accumulates histogram
     * 4. Evaluates SAY likelihood
     * 5. Optionally adds prior terms
     *
     * @param params Fit parameters [NUM_FIT_PARAMS]
     * @param includePrior Include prior terms in likelihood
     * @return Negative log-likelihood value
     */
    double evaluateLikelihood(
        const std::vector<double>& params,
        bool includePrior = true
    );

    /**
     * @brief Compute negative log-likelihood with gradient
     *
     * Uses GPU autodiff to compute both value and gradient in one pass.
     *
     * @param params Fit parameters [NUM_FIT_PARAMS]
     * @param gradient Output: gradient with respect to each parameter [NUM_FIT_PARAMS]
     * @param includePrior Include prior terms
     * @return Negative log-likelihood value
     */
    double evaluateLikelihoodWithGradient(
        const std::vector<double>& params,
        std::vector<double>& gradient,
        bool includePrior = true
    );

    //==========================================================================
    // Histogram Access
    //==========================================================================

    /**
     * @brief Get current MC expectation histogram
     *
     * Returns the weighted histogram from the last likelihood evaluation.
     *
     * @param histogram Output: expected counts per bin [totalBins]
     */
    void getExpectationHistogram(std::vector<double>& histogram) const;

    /**
     * @brief Get sum of squared weights histogram
     *
     * For SAY likelihood uncertainty calculation.
     *
     * @param histogram Output: sum of w^2 per bin [totalBins]
     */
    void getWeightSquaredHistogram(std::vector<double>& histogram) const;

    /**
     * @brief Get per-event weights from last likelihood evaluation
     *
     * @param weights Output: per-event weight [numEvents]
     */
    void getEventWeights(std::vector<double>& weights) const;

    //==========================================================================
    // Utility Functions
    //==========================================================================

    /**
     * @brief Check if GPU is initialized and ready
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Get number of events on GPU
     */
    size_t getNumEvents() const;

    /**
     * @brief Get number of histogram bins
     */
    int getNumBins() const;

    /**
     * @brief Get GPU device info
     */
    GPUDeviceInfo getDeviceInfo() const;

    /**
     * @brief Synchronize all GPU operations
     */
    void synchronize();

    /**
     * @brief Reset GPU state (clear histograms, etc.)
     */
    void reset();

    /**
     * @brief Get timing statistics from last evaluation
     */
    struct TimingStats {
        double paramTransferMs = 0.0;
        double weightComputeMs = 0.0;
        double histogramMs = 0.0;
        double likelihoodMs = 0.0;
        double totalMs = 0.0;
    };
    TimingStats getLastTimingStats() const { return lastTiming_; }

private:
    // Configuration
    GPUAcceleratorConfig config_;
    HistogramConfig histConfig_;
    bool initialized_ = false;

    // Event data
    std::unique_ptr<GPUEventDataManager> eventManager_;

    // Spline management
    GPUSplineManager splineManager_;
    std::unordered_map<std::string, int> splineNameToIndex_;

    // Device memory for histograms
    DevicePtr<double> d_binSums_;         // MC weight sums [totalBins]
    DevicePtr<double> d_binSqSums_;       // MC weight squared sums [totalBins]
    DevicePtr<double> d_dataCount_;       // Observed data [totalBins]

    // Device memory for weights
    DevicePtr<double> d_weights_;         // Per-event weights [numEvents]
    DevicePtr<double> d_weightsSquared_;  // Per-event w^2 [numEvents]

    // Device memory for parameters
    DevicePtr<double> d_params_;          // Fit parameters [NUM_FIT_PARAMS]

    // Prior data on device
    DevicePtr<double> d_priorMeans_;
    DevicePtr<double> d_priorSigmas_;
    DevicePtr<int> d_priorFlags_;
    bool priorsConfigured_ = false;

    // N-dimensional correlated Gaussian priors (computed on CPU)
    std::vector<NDPriorConfig> ndPriors_;

    // Pinned host memory for async transfers
    PinnedPtr<double> h_params_;

    // Adjoint method buffers
    DevicePtr<double> d_adjoint_wsum_;     // [totalBins] dSAY/d(w_sum)
    DevicePtr<double> d_adjoint_w2sum_;    // [totalBins] dSAY/d(w2_sum)
    DevicePtr<double> d_gradient_;         // [NUM_FIT_PARAMS] gradient accumulator
    PinnedPtr<double> h_gradient_;         // [NUM_FIT_PARAMS] pinned host copy

    // Gradient intermediate buffer (per-event cached scalars for analytic derivatives)
    DevicePtr<double> d_gradIntermediate_; // [26 * numEvents]

    // Host copies of prior config for CPU gradient computation
    std::vector<double> h_priorMeans_;
    std::vector<double> h_priorSigmas_;
    std::vector<int> h_priorFlags_;

    // CUDA streams
    std::vector<CudaStream> streams_;

    // Timing
    TimingStats lastTiming_;

    // Spline lookup for kernels
    GPUSplineLookup splineLookup_;

    // Whether cached basis is valid (all DOM eff / hole ice splines share knots for dims 0&1)
    bool basisCacheValid_ = false;

    // Internal helper methods
    void allocateDeviceMemory();
    void computeWeights(cudaStream_t stream);
    void computeWeightsWithGradient(cudaStream_t stream);
    void accumulateHistogram(cudaStream_t stream);
    double computeLikelihood(bool includePrior, cudaStream_t stream);
    void addPriorGradient(std::vector<double>& gradient);
};

//==============================================================================
// Template Implementation
//==============================================================================

template<typename EventContainer>
void GPUFitAccelerator::initialize(
    const EventContainer& events,
    const HistogramConfig& histConfig
) {
    histConfig_ = histConfig;

    // Create and populate event manager
    eventManager_ = std::make_unique<GPUEventDataManager>();

    // Upload events (AoS to SoA transformation happens here)
    eventManager_->uploadFromCPU(events, nullptr);

    // Compute bin indices for histogram accumulation
    eventManager_->computeBinIndices(
        histConfig.energyEdges,
        histConfig.zenithEdges,
        histConfig.nBinsTopology
    );

    // Allocate device memory for histograms and weights
    allocateDeviceMemory();

    initialized_ = true;
}

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA

#endif // GOLLUMFIT_GPU_FIT_ACCELERATOR_H
