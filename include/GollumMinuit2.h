#ifndef GOLLUM_MINUIT2_H
#define GOLLUM_MINUIT2_H

#ifdef GOLLUMFIT_USE_MINUIT2

#include <Minuit2/FCNGradientBase.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/MnUserCovariance.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/MnPrint.h>

#include <vector>

namespace gollumfit {

// Forward declaration
class GollumFit;

/// Minuit2 function adapter for GollumFit.
/// Provides function value and analytical gradient to MIGRAD.
/// Caches the last evaluation to avoid redundant spline computations
/// when Minuit2 calls operator() and Gradient() at the same point.
class GollumMinuit2FCN : public ROOT::Minuit2::FCNGradientBase {
    const GollumFit& fitter_;

    // Cache to avoid double-evaluating value + gradient
    mutable std::vector<double> cachedParams_;
    mutable double cachedNLL_ = 0;
    mutable std::vector<double> cachedGrad_;
    mutable bool cacheValid_ = false;

    void ensureEvaluated(const std::vector<double>& params) const;

public:
    explicit GollumMinuit2FCN(const GollumFit& f) : fitter_(f) {}

    /// Function value (negative log-likelihood including prior)
    double operator()(const std::vector<double>& params) const override;

    /// Analytical gradient from the adjoint method
    std::vector<double> Gradient(const std::vector<double>& params) const override;

    /// For NLL minimization, Up() = 0.5 gives 1-sigma errors
    double Up() const override { return 0.5; }

    /// Gradients are in external (user) parameter space
    ROOT::Minuit2::GradientParameterSpace gradParameterSpace() const override {
        return ROOT::Minuit2::GradientParameterSpace::External;
    }
};

} // namespace gollumfit

#endif // GOLLUMFIT_USE_MINUIT2
#endif // GOLLUM_MINUIT2_H
