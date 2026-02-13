/**
 * @file GPUSplineTable.h
 * @brief GPU-compatible B-spline table for systematic corrections.
 *
 * This file provides GPU-compatible B-spline evaluation that matches the
 * photospline library interface. The spline tables are transferred to GPU
 * memory once at initialization and can be efficiently evaluated during
 * kernel execution.
 *
 * Supported spline types:
 * - 2D splines (e.g., atmospheric density uncertainty)
 * - 3D splines (e.g., DOM efficiency, hole ice, attenuation)
 */

#ifndef GOLLUMFIT_GPU_SPLINE_TABLE_H
#define GOLLUMFIT_GPU_SPLINE_TABLE_H

#ifdef GOLLUMFIT_USE_CUDA

#include "GPUCommon.h"
#include <vector>
#include <memory>

namespace gollumfit {
namespace gpu {

//==============================================================================
// GPU Spline Table Structure
//==============================================================================

/**
 * @brief GPU-resident B-spline table structure
 *
 * This structure contains all data needed to evaluate B-splines on the GPU.
 * It supports up to 3 dimensions (matching the splines used in GollumFit).
 */
struct GPUSplineTable {
    // Spline metadata
    int ndim;                    ///< Number of dimensions (1-3)
    int order[3];                ///< B-spline order per dimension
    int nknots[3];               ///< Number of knots per dimension
    int ncoeffs;                 ///< Total number of coefficients

    // Knot vectors (device pointers)
    double* knots[3];            ///< Knot vectors per dimension

    // Coefficients (device pointer, flattened)
    double* coeffs;              ///< Spline coefficients

    // Coefficient strides for indexing
    int strides[3];              ///< Strides for coefficient array

    // Extents (for bounds checking)
    double extents_min[3];       ///< Minimum valid coordinate per dim
    double extents_max[3];       ///< Maximum valid coordinate per dim

    /**
     * @brief Check if coordinates are within valid range
     *
     * Matches photospline's searchcenters() outer bounds: (knots[0], knots[nknots-1]]
     * For edge-region coordinates (between knots[0] and knots[order], or between
     * knots[naxes] and knots[nknots-1]), findKnotSpan clamps the span just like
     * photospline's searchcenters() does, producing identical evaluation results.
     */
    __device__ bool inBounds(const double* coords) const {
        for (int d = 0; d < ndim; ++d) {
            if (coords[d] <= knots[d][0] || coords[d] > knots[d][nknots[d] - 1]) {
                return false;
            }
        }
        return true;
    }
};

//==============================================================================
// GPU Spline Manager
//==============================================================================

/**
 * @brief Manager class for GPU spline tables
 *
 * Handles allocation, transfer, and cleanup of spline tables on the GPU.
 */
class GPUSplineManager {
public:
    GPUSplineManager();
    ~GPUSplineManager();

    // Non-copyable
    GPUSplineManager(const GPUSplineManager&) = delete;
    GPUSplineManager& operator=(const GPUSplineManager&) = delete;

    // Movable
    GPUSplineManager(GPUSplineManager&& other) noexcept;
    GPUSplineManager& operator=(GPUSplineManager&& other) noexcept;

    /**
     * @brief Upload a spline table from CPU (photospline format)
     *
     * @param ndim Number of dimensions
     * @param orders B-spline orders per dimension
     * @param knots Knot vectors per dimension
     * @param coeffs Coefficient array (row-major order)
     * @param nknots Number of knots per dimension
     * @return Index of the uploaded spline table
     */
    int uploadSpline(
        int ndim,
        const int* orders,
        const std::vector<std::vector<double>>& knots,
        const std::vector<double>& coeffs,
        const int* nknots
    );

    /**
     * @brief Get device pointer to a spline table
     */
    const GPUSplineTable* getDeviceSpline(int index) const;

    /**
     * @brief Get number of uploaded splines
     */
    size_t getNumSplines() const { return deviceSplines_.size(); }

    /**
     * @brief Clear all splines from GPU memory
     */
    void clear();

private:
    void freeSpline(GPUSplineTable& spline);

    std::vector<GPUSplineTable> hostSplines_;    ///< Host copies of metadata
    std::vector<GPUSplineTable*> deviceSplines_; ///< Device pointers to full structures
};

//==============================================================================
// Spline Lookup Structure for Event Weighting
//==============================================================================

/**
 * @brief Enum for flux components (matching gollumfit::FluxComponent)
 */
enum GPUFluxComponent {
    GPU_FLUX_CONV = 0,
    GPU_FLUX_PROMPT = 1,
    GPU_FLUX_ASTRO = 2,
    GPU_NUM_FLUX_COMPONENTS = 3
};

/**
 * @brief Enum for topologies (matching gollumfit::Topology)
 */
enum GPUTopology {
    GPU_TOPO_CASCADE = 0,
    GPU_TOPO_TRACK = 1,
    GPU_NUM_TOPOLOGIES = 2
};

/**
 * @brief Enum for particle types used in attenuation spline lookup
 *
 * Maps LW::ParticleType values to compact indices:
 *   NuMu (14) → 0, NuMuBar (-14) → 1, NuTau (16) → 2, NuTauBar (-16) → 3
 */
enum GPUParticleType {
    GPU_PTYPE_NUMU = 0,
    GPU_PTYPE_NUMUBAR = 1,
    GPU_PTYPE_NUTAU = 2,
    GPU_PTYPE_NUTAUBAR = 3,
    GPU_NUM_PARTICLE_TYPES = 4
};

/**
 * @brief Structure holding spline pointers for event weighting
 *
 * This structure organizes splines by flux component and topology for
 * efficient lookup during kernel execution. All pointers are device pointers.
 */
struct GPUSplineLookup {
    // DOM efficiency splines [flux_component][topology]
    const GPUSplineTable* domEffSplines[GPU_NUM_FLUX_COMPONENTS][GPU_NUM_TOPOLOGIES];

    // Hole ice splines [flux_component][topology]
    const GPUSplineTable* holeIceSplines[GPU_NUM_FLUX_COMPONENTS][GPU_NUM_TOPOLOGIES];

    // Attenuation splines [flux_component][particle_type]
    // Indexed by (GPUFluxComponent, GPUParticleType)
    // Only NuMu, NuMuBar, NuTau, NuTauBar have attenuation splines
    const GPUSplineTable* attenSplines[GPU_NUM_FLUX_COMPONENTS][GPU_NUM_PARTICLE_TYPES];

    // Reference values for corrections
    double domEffReference;      // Typically 1.27
    double holeIceReference;     // Typically -1.0

    // Flag to indicate if splines are available
    bool hasSplines;

    /**
     * @brief Initialize all pointers to nullptr
     */
    __host__ void clear() {
        for (int f = 0; f < GPU_NUM_FLUX_COMPONENTS; ++f) {
            for (int t = 0; t < GPU_NUM_TOPOLOGIES; ++t) {
                domEffSplines[f][t] = nullptr;
                holeIceSplines[f][t] = nullptr;
            }
            for (int p = 0; p < GPU_NUM_PARTICLE_TYPES; ++p) {
                attenSplines[f][p] = nullptr;
            }
        }
        domEffReference = 1.27;
        holeIceReference = -1.0;
        hasSplines = false;
    }
};

/**
 * @brief Map LW::ParticleType int value to GPUParticleType index
 *
 * Returns -1 for particle types without attenuation splines (NuE, NuEBar).
 *
 * @param primaryType LW::ParticleType as int32_t (e.g. 14=NuMu, -14=NuMuBar)
 * @return GPU particle type index, or -1 if no attenuation spline exists
 */
__device__ __host__ inline int mapParticleTypeToGPU(int32_t primaryType) {
    switch (primaryType) {
        case 14:  return GPU_PTYPE_NUMU;      // NuMu
        case -14: return GPU_PTYPE_NUMUBAR;   // NuMuBar
        case 16:  return GPU_PTYPE_NUTAU;     // NuTau
        case -16: return GPU_PTYPE_NUTAUBAR;  // NuTauBar
        default:  return -1;                  // No attenuation spline (NuE, NuEBar, etc.)
    }
}

//==============================================================================
// Device-side B-spline Evaluation
//==============================================================================

/**
 * @brief Find the knot span containing a coordinate
 *
 * Binary search to find i such that knots[i] <= x < knots[i+1]
 *
 * NOTE: 'order' here is the B-spline DEGREE (as stored by photospline's
 * get_order()). The number of non-zero basis functions is degree + 1.
 *
 * @param x Coordinate value
 * @param knots Knot vector
 * @param nknots Number of knots
 * @param order B-spline degree (photospline convention)
 * @return Knot span index
 */
__device__ inline int findKnotSpan(double x, const double* knots, int nknots, int order) {
    // n_centers = number of coefficients = nknots - degree - 1
    int n = nknots - order - 1;
    // Upper boundary: clamp to last valid span (matching photospline)
    if (x >= knots[n]) return n - 1;
    // Lower boundary: first valid span
    if (x <= knots[order]) return order;

    // Binary search
    int low = order;
    int high = n;
    int mid = (low + high) / 2;

    while (x < knots[mid] || x >= knots[mid + 1]) {
        if (x < knots[mid]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }

    return mid;
}

/**
 * @brief Evaluate B-spline basis functions using de Boor's algorithm
 *
 * Computes non-zero basis functions at point x.
 * Generates (degree + 1) basis function values.
 *
 * Matches photospline's bsplvb_simple() including edge-region corrections:
 * when span is at the boundary (clamped by findKnotSpan), the actual knot
 * interval may differ from span. The algorithm adjusts for this and shifts
 * basis functions accordingly, zeroing out unsupported ones.
 *
 * NOTE: 'order' here is the B-spline DEGREE (photospline convention).
 * The loop runs j = 1..degree, producing degree+1 basis function values.
 *
 * @param x Coordinate value
 * @param knots Knot vector
 * @param nknots Number of knots in this dimension
 * @param span Knot span index (from findKnotSpan, may be clamped)
 * @param order B-spline degree (photospline convention)
 * @param basis Output array for basis function values [order+1 elements]
 */
__device__ inline void evaluateBasis(
    double x,
    const double* knots,
    int nknots,
    int span,
    int order,
    double* basis
) {
    // Work arrays (allocated in registers for small orders)
    double dleft[8];   // Max degree 7
    double dright[8];

    // Edge-region correction (matching photospline bsplvb_simple):
    // When span is clamped to the boundary by findKnotSpan, adjust to
    // the actual knot interval containing x before running de Boor.
    int left = span;
    int degree = order + 1;  // photospline convention: degree = order + 1

    if (left == degree - 1) {
        // Left edge: move left down to actual interval
        while (left >= 0 && x < knots[left])
            left--;
    } else if (left == nknots - degree - 1) {
        // Right edge: move left up to actual interval
        while (left < nknots - 1 && x > knots[left + 1])
            left++;
    }

    // Standard de Boor algorithm with adjusted 'left'
    // Clamp knot accesses to [0, nknots-1] to avoid OOB reads in edge regions.
    // The contaminated basis values get zeroed by the boundary shift below.
    basis[0] = 1.0;

    for (int j = 1; j <= order; ++j) {
        int li = left + 1 - j;
        int ri = left + j;
        dleft[j] = x - knots[li >= 0 ? li : 0];
        dright[j] = knots[ri < nknots ? ri : nknots - 1] - x;
        double saved = 0.0;

        for (int r = 0; r < j; ++r) {
            double temp = basis[r] / (dright[r + 1] + dleft[j - r]);
            basis[r] = saved + dright[r + 1] * temp;
            saved = dleft[j - r] * temp;
        }
        basis[j] = saved;
    }

    // Boundary correction: shift basis functions to account for the
    // difference between the original span and the actual left.
    // Matches photospline bsplvb_simple lines 83-93.
    int shift;
    if ((shift = degree - 1 - left) > 0) {
        // Left boundary: only (left+1) basis functions are valid
        for (int j = 0; j < left + 1; j++)
            basis[j] = basis[j + shift];
        for (int j = left + 1; j < degree; j++)
            basis[j] = 0.0;
    } else if ((shift = left + degree + 1 - nknots) > 0) {
        // Right boundary: shift valid functions to the end
        for (int j = degree - 1; j > shift - 1; j--)
            basis[j] = basis[j - shift];
        for (int j = shift - 1; j >= 0; j--)
            basis[j] = 0.0;
    }
}

/**
 * @brief Evaluate a 2D B-spline at given coordinates
 *
 * @param spline Spline table structure
 * @param x Coordinate in dimension 0
 * @param y Coordinate in dimension 1
 * @return Interpolated value
 */
__device__ inline double evaluateSpline2D(
    const GPUSplineTable& spline,
    double x,
    double y
) {
    double coords[2] = {x, y};

    // Check bounds
    if (!spline.inBounds(coords)) {
        return 0.0;  // Or could return NaN
    }

    // Find knot spans
    int spans[2];
    spans[0] = findKnotSpan(x, spline.knots[0], spline.nknots[0], spline.order[0]);
    spans[1] = findKnotSpan(y, spline.knots[1], spline.nknots[1], spline.order[1]);

    // Evaluate basis functions
    double basis0[8], basis1[8];
    evaluateBasis(x, spline.knots[0], spline.nknots[0], spans[0], spline.order[0], basis0);
    evaluateBasis(y, spline.knots[1], spline.nknots[1], spans[1], spline.order[1], basis1);

    // Sum over tensor product of basis functions
    // Loop bounds: 0..degree (inclusive) = degree+1 basis functions per dimension
    double result = 0.0;
    for (int i = 0; i <= spline.order[0]; ++i) {
        int idx0 = spans[0] - spline.order[0] + i;
        for (int j = 0; j <= spline.order[1]; ++j) {
            int idx1 = spans[1] - spline.order[1] + j;
            int coeff_idx = idx0 * spline.strides[0] + idx1 * spline.strides[1];
            result += basis0[i] * basis1[j] * spline.coeffs[coeff_idx];
        }
    }

    return result;
}

/**
 * @brief Evaluate a 3D B-spline at given coordinates
 *
 * @param spline Spline table structure
 * @param x Coordinate in dimension 0
 * @param y Coordinate in dimension 1
 * @param z Coordinate in dimension 2
 * @return Interpolated value
 */
__device__ inline double evaluateSpline3D(
    const GPUSplineTable& spline,
    double x,
    double y,
    double z
) {
    double coords[3] = {x, y, z};

    // Check bounds
    if (!spline.inBounds(coords)) {
        return 0.0;
    }

    // Find knot spans
    int spans[3];
    spans[0] = findKnotSpan(x, spline.knots[0], spline.nknots[0], spline.order[0]);
    spans[1] = findKnotSpan(y, spline.knots[1], spline.nknots[1], spline.order[1]);
    spans[2] = findKnotSpan(z, spline.knots[2], spline.nknots[2], spline.order[2]);

    // Evaluate basis functions
    double basis0[8], basis1[8], basis2[8];
    evaluateBasis(x, spline.knots[0], spline.nknots[0], spans[0], spline.order[0], basis0);
    evaluateBasis(y, spline.knots[1], spline.nknots[1], spans[1], spline.order[1], basis1);
    evaluateBasis(z, spline.knots[2], spline.nknots[2], spans[2], spline.order[2], basis2);

    // Sum over tensor product of basis functions
    // Loop bounds: 0..degree (inclusive) = degree+1 basis functions per dimension
    double result = 0.0;
    for (int i = 0; i <= spline.order[0]; ++i) {
        int idx0 = spans[0] - spline.order[0] + i;
        for (int j = 0; j <= spline.order[1]; ++j) {
            int idx1 = spans[1] - spline.order[1] + j;
            for (int k = 0; k <= spline.order[2]; ++k) {
                int idx2 = spans[2] - spline.order[2] + k;
                int coeff_idx = idx0 * spline.strides[0] +
                               idx1 * spline.strides[1] +
                               idx2 * spline.strides[2];
                result += basis0[i] * basis1[j] * basis2[k] * spline.coeffs[coeff_idx];
            }
        }
    }

    return result;
}

/**
 * @brief Evaluate B-spline basis function derivatives analytically
 *
 * Uses the B-spline derivative recurrence:
 *   dN_{i,p}/dx = p * [ N_{i,p-1}(x) / (t_{i+p} - t_i)
 *                      - N_{i+1,p-1}(x) / (t_{i+p+1} - t_{i+1}) ]
 *
 * We evaluate order-1 basis functions at the same span, then apply the formula.
 *
 * @param x Coordinate value
 * @param knots Knot vector
 * @param span Knot span index (from findKnotSpan at full order)
 * @param order B-spline degree (photospline convention)
 * @param dbasis Output: derivatives of the (order+1) non-zero basis functions
 */
__device__ inline void evaluateBasisDerivative(
    double x,
    const double* knots,
    int nknots,
    int span,
    int order,
    double* dbasis
) {
    if (order == 0) {
        // Degree-0 splines are piecewise constant => derivative is zero
        dbasis[0] = 0.0;
        return;
    }

    // Evaluate basis functions of degree (order-1) at the same span.
    // At knot span 'span', the non-zero degree-(order-1) basis functions are
    // N_{span-order+1, order-1} through N_{span, order-1}, which is (order) functions.
    // But we also need N_{span+1, order-1} for the last derivative term.
    // We evaluate (order+1) lower-order basis functions by adjusting the span.
    //
    // Actually, the standard approach: evaluate (order) lower-order basis functions
    // at the original span, giving N_{span-(order-1), order-1} ... N_{span, order-1}.
    // These are stored as lower[0..order-1].
    double lower[8];
    evaluateBasis(x, knots, nknots, span, order - 1, lower);

    // The degree-p basis functions non-zero at span are:
    //   N_{span-p, p}, N_{span-p+1, p}, ..., N_{span, p}  (p+1 functions)
    // Their derivatives use degree-(p-1) basis functions:
    //   dN_{span-p+r, p}/dx = p * [ N_{span-p+r, p-1} / (t_{span+r} - t_{span-p+r})
    //                              - N_{span-p+r+1, p-1} / (t_{span+r+1} - t_{span-p+r+1}) ]
    //
    // The degree-(p-1) basis functions non-zero at span are:
    //   N_{span-(p-1), p-1}, ..., N_{span, p-1}  (p functions)
    // stored as lower[0] = N_{span-(p-1), p-1}, ..., lower[p-1] = N_{span, p-1}
    //
    // For derivative of N_{span-p+r, p}: we need
    //   lower index for N_{span-p+r, p-1}: this is index (r-1)
    //   lower index for N_{span-p+r+1, p-1}: this is index r

    for (int r = 0; r <= order; ++r) {
        double term1 = 0.0;
        double term2 = 0.0;

        // First term: N_{span-order+r, order-1} / (t_{span+r} - t_{span-order+r})
        int idx1 = r - 1;  // index into lower[]
        if (idx1 >= 0 && idx1 < order) {
            double denom = knots[span + r] - knots[span - order + r];
            if (denom > 0.0) {
                term1 = lower[idx1] / denom;
            }
        }

        // Second term: N_{span-order+r+1, order-1} / (t_{span+r+1} - t_{span-order+r+1})
        int idx2 = r;  // index into lower[]
        if (idx2 >= 0 && idx2 < order) {
            double denom = knots[span + r + 1] - knots[span - order + r + 1];
            if (denom > 0.0) {
                term2 = lower[idx2] / denom;
            }
        }

        dbasis[r] = static_cast<double>(order) * (term1 - term2);
    }
}

/**
 * @brief Evaluate 3D spline value and derivative w.r.t. dimension 2 (z)
 *
 * Optimized for the common case in gradient computation where only the
 * 3rd coordinate depends on a fit parameter (DOM eff, hole ice, attenuation).
 * Uses analytic basis derivatives via the B-spline recurrence formula.
 *
 * @param spline Spline table structure
 * @param x Coordinate in dimension 0 (e.g. log10Energy)
 * @param y Coordinate in dimension 1 (e.g. cosZenith)
 * @param z Coordinate in dimension 2 (e.g. domEfficiency)
 * @param value Output: spline value
 * @param dvalue_dz Output: derivative with respect to z
 */
__device__ inline void evaluateSpline3DValueAndDerivZ(
    const GPUSplineTable& spline,
    double x,
    double y,
    double z,
    double& value,
    double& dvalue_dz
) {
    double coords[3] = {x, y, z};

    if (!spline.inBounds(coords)) {
        value = 0.0;
        dvalue_dz = 0.0;
        return;
    }

    // Find knot spans
    int spans[3];
    spans[0] = findKnotSpan(x, spline.knots[0], spline.nknots[0], spline.order[0]);
    spans[1] = findKnotSpan(y, spline.knots[1], spline.nknots[1], spline.order[1]);
    spans[2] = findKnotSpan(z, spline.knots[2], spline.nknots[2], spline.order[2]);

    // Evaluate basis functions for dims 0 and 1
    double basis0[8], basis1[8], basis2[8];
    evaluateBasis(x, spline.knots[0], spline.nknots[0], spans[0], spline.order[0], basis0);
    evaluateBasis(y, spline.knots[1], spline.nknots[1], spans[1], spline.order[1], basis1);
    evaluateBasis(z, spline.knots[2], spline.nknots[2], spans[2], spline.order[2], basis2);

    // Analytic basis derivatives for dimension 2
    double dbasis2[8];
    evaluateBasisDerivative(z, spline.knots[2], spline.nknots[2], spans[2], spline.order[2], dbasis2);

    // Compute value and derivative simultaneously
    value = 0.0;
    dvalue_dz = 0.0;

    for (int i = 0; i <= spline.order[0]; ++i) {
        int idx0 = spans[0] - spline.order[0] + i;
        for (int j = 0; j <= spline.order[1]; ++j) {
            int idx1 = spans[1] - spline.order[1] + j;
            double b01 = basis0[i] * basis1[j];
            for (int k = 0; k <= spline.order[2]; ++k) {
                int idx2 = spans[2] - spline.order[2] + k;
                int coeff_idx = idx0 * spline.strides[0] +
                               idx1 * spline.strides[1] +
                               idx2 * spline.strides[2];
                double c = spline.coeffs[coeff_idx];

                value += b01 * basis2[k] * c;
                dvalue_dz += b01 * dbasis2[k] * c;
            }
        }
    }
}

/**
 * @brief Evaluate 3D spline and its derivative with respect to dimension d
 *
 * Used for automatic differentiation through spline evaluations.
 * Uses analytic basis derivatives via B-spline recurrence formula.
 *
 * @param spline Spline table structure
 * @param x Coordinate in dimension 0
 * @param y Coordinate in dimension 1
 * @param z Coordinate in dimension 2
 * @param dim Dimension with respect to which to compute derivative (0-2)
 * @param value Output: spline value
 * @param deriv Output: derivative with respect to specified dimension
 */
__device__ inline void evaluateSpline3DWithDerivative(
    const GPUSplineTable& spline,
    double x,
    double y,
    double z,
    int dim,
    double& value,
    double& deriv
) {
    double coords[3] = {x, y, z};

    if (!spline.inBounds(coords)) {
        value = 0.0;
        deriv = 0.0;
        return;
    }

    // Find knot spans
    int spans[3];
    spans[0] = findKnotSpan(x, spline.knots[0], spline.nknots[0], spline.order[0]);
    spans[1] = findKnotSpan(y, spline.knots[1], spline.nknots[1], spline.order[1]);
    spans[2] = findKnotSpan(z, spline.knots[2], spline.nknots[2], spline.order[2]);

    // Evaluate basis functions
    double basis0[8], basis1[8], basis2[8];
    evaluateBasis(x, spline.knots[0], spline.nknots[0], spans[0], spline.order[0], basis0);
    evaluateBasis(y, spline.knots[1], spline.nknots[1], spans[1], spline.order[1], basis1);
    evaluateBasis(z, spline.knots[2], spline.nknots[2], spans[2], spline.order[2], basis2);

    // Analytic basis derivatives for the requested dimension
    double basis_deriv[8];
    if (dim == 0) {
        evaluateBasisDerivative(x, spline.knots[0], spline.nknots[0], spans[0], spline.order[0], basis_deriv);
    } else if (dim == 1) {
        evaluateBasisDerivative(y, spline.knots[1], spline.nknots[1], spans[1], spline.order[1], basis_deriv);
    } else {
        evaluateBasisDerivative(z, spline.knots[2], spline.nknots[2], spans[2], spline.order[2], basis_deriv);
    }

    // Compute value and derivative
    value = 0.0;
    deriv = 0.0;

    for (int i = 0; i <= spline.order[0]; ++i) {
        int idx0 = spans[0] - spline.order[0] + i;
        double b0 = basis0[i];
        double db0 = (dim == 0) ? basis_deriv[i] : 0.0;

        for (int j = 0; j <= spline.order[1]; ++j) {
            int idx1 = spans[1] - spline.order[1] + j;
            double b1 = basis1[j];
            double db1 = (dim == 1) ? basis_deriv[j] : 0.0;

            for (int k = 0; k <= spline.order[2]; ++k) {
                int idx2 = spans[2] - spline.order[2] + k;
                double b2 = basis2[k];
                double db2 = (dim == 2) ? basis_deriv[k] : 0.0;

                int coeff_idx = idx0 * spline.strides[0] +
                               idx1 * spline.strides[1] +
                               idx2 * spline.strides[2];
                double c = spline.coeffs[coeff_idx];

                value += b0 * b1 * b2 * c;

                // Product rule for derivative
                if (dim == 0) {
                    deriv += db0 * b1 * b2 * c;
                } else if (dim == 1) {
                    deriv += b0 * db1 * b2 * c;
                } else {
                    deriv += b0 * b1 * db2 * c;
                }
            }
        }
    }
}

} // namespace gpu
} // namespace gollumfit

#endif // GOLLUMFIT_USE_CUDA

#endif // GOLLUMFIT_GPU_SPLINE_TABLE_H
