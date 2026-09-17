"""
Classifier-agnostic Bayes-risk bounds via Gaussian and nonparametric
class-conditional density estimation.

WHY THIS EXISTS (see paper section "A Data-Intrinsic Bayes-Risk Benchmark"):
Every existing evaluation of RGB-T fusion reports the error of one trained
model. That number is confounded by architecture choice, training budget,
and optimization luck. The Bayes risk is a property of the DATA, not of
any model: it is the error rate of the *optimal* classifier for that
data, and classical statistical decision theory gives tight, closed-form
or cheaply-estimable
bounds on it that require no gradient descent at all.

Implements two independent estimators so results can be cross-checked
against each other (parametric vs. nonparametric assumption):

1. Gaussian Bhattacharyya bound (closed form). For class-conditional
   features modeled as N(mu0, Sigma0), N(mu1, Sigma1) with priors pi0, pi1:

       B = (1/8)(mu1-mu0)^T [(Sigma0+Sigma1)/2]^{-1} (mu1-mu0)
           + (1/2) ln( det((Sigma0+Sigma1)/2) / sqrt(det(Sigma0)*det(Sigma1)) )

       Bhattacharyya coefficient: BC = exp(-B)
       Bayes error bound:          P_e <= sqrt(pi0*pi1) * BC

   (Bhattacharyya 1943; Fukunaga, "Introduction to Statistical Pattern
   Recognition," 1990, Ch. 3 -- this exact closed form is standard
   textbook material, not a novel derivation; the contribution here is
   applying it as a dataset-benchmarking tool, not the formula itself.)

2. Nonparametric Chernoff/Bhattacharyya via kernel density estimation, as
   a cross-check that doesn't assume Gaussianity: BC = integral of
   sqrt(p(x|0) p(x|1)) dx, estimated by Monte Carlo integration against a
   KDE fit to each class's real samples.

3. Mutual information I(Y;X) via Monte Carlo entropy estimation on the
   fitted Gaussian mixture: I(Y;X) = H(X) - sum_y pi_y H(X|Y=y). H(X|Y=y)
   is closed-form for a Gaussian; H(X) (the mixture) is estimated by
   sampling from the mixture and averaging -log(mixture density) at those
   samples (a standard, unbiased-in-the-large-sample-limit technique).
   This is what instantiates the data-processing-inequality argument
   numerically against the real measured misalignment distribution.
"""

import numpy as np
from scipy.stats import gaussian_kde


def fit_gaussian(features):
    """features: (N, D) array. Returns (mean, cov) with a small ridge added
    to cov for numerical stability when N is not much larger than D."""
    mean = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov = cov + np.eye(cov.shape[0]) * 1e-6 * np.trace(cov) / cov.shape[0]
    return mean, cov


def bhattacharyya_gaussian(mean0, cov0, mean1, cov1):
    """Closed-form Bhattacharyya distance and coefficient between two
    multivariate Gaussians."""
    cov_avg = (cov0 + cov1) / 2
    diff = mean1 - mean0
    inv_cov_avg = np.linalg.inv(cov_avg)

    term1 = 0.125 * diff.T @ inv_cov_avg @ diff
    sign0, logdet0 = np.linalg.slogdet(cov0)
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
    term2 = 0.5 * (logdet_avg - 0.5 * (logdet0 + logdet1))

    b_dist = float(term1 + term2)
    bc = float(np.exp(-b_dist))
    return b_dist, bc


def bayes_error_bound_gaussian(features0, features1, prior0=0.5, prior1=0.5):
    """Returns dict with the Gaussian Bhattacharyya distance, coefficient,
    and the resulting Bayes error upper bound P_e <= sqrt(pi0*pi1)*BC."""
    mean0, cov0 = fit_gaussian(features0)
    mean1, cov1 = fit_gaussian(features1)
    b_dist, bc = bhattacharyya_gaussian(mean0, cov0, mean1, cov1)
    bound = float(np.sqrt(prior0 * prior1) * bc)
    return {"b_distance": b_dist, "bhattacharyya_coefficient": bc, "bayes_error_bound": bound,
            "n0": len(features0), "n1": len(features1)}


def chernoff_gaussian(mean0, cov0, mean1, cov1, s):
    """Generalized Chernoff distance between two Gaussians at parameter
    s in (0,1) (Fukunaga 1990, Ch. 3; Chernoff 1952). s=0.5 recovers the
    Bhattacharyya distance exactly (up to the prior prefactor)."""
    cov_s = s * cov1 + (1 - s) * cov0
    diff = mean1 - mean0
    inv_cov_s = np.linalg.inv(cov_s)

    term1 = (s * (1 - s) / 2) * diff.T @ inv_cov_s @ diff
    sign_s, logdet_s = np.linalg.slogdet(cov_s)
    sign0, logdet0 = np.linalg.slogdet(cov0)
    sign1, logdet1 = np.linalg.slogdet(cov1)
    term2 = 0.5 * (logdet_s - (1 - s) * logdet0 - s * logdet1)

    return float(term1 + term2)


def optimized_chernoff_bound_gaussian(features0, features1, prior0=0.5, prior1=0.5, n_grid=99):
    """Bayes error bound P_e <= pi0^s * pi1^(1-s) * exp(-mu(s)), minimized
    over s in (0,1) on a fine grid. This is the tightest bound in the
    Chernoff family; the closed-form Bhattacharyya bound
    (bayes_error_bound_gaussian) is the single point s=0.5 in this same
    family, so the optimized bound is guaranteed at least as tight."""
    mean0, cov0 = fit_gaussian(features0)
    mean1, cov1 = fit_gaussian(features1)

    s_grid = np.linspace(0.01, 0.99, n_grid)
    best_bound, best_s, best_mu = None, None, None
    for s in s_grid:
        mu_s = chernoff_gaussian(mean0, cov0, mean1, cov1, s)
        bound = float((prior0 ** (1 - s)) * (prior1 ** s) * np.exp(-mu_s))
        if best_bound is None or bound < best_bound:
            best_bound, best_s, best_mu = bound, float(s), mu_s
    return {"bayes_error_bound": best_bound, "optimal_s": best_s,
            "chernoff_distance": best_mu, "n0": len(features0), "n1": len(features1)}


def bayes_error_bound_kde(features0, features1, prior0=0.5, prior1=0.5, n_mc=5000, seed=0):
    """Nonparametric cross-check: fit a KDE to each class, estimate the
    Bhattacharyya coefficient integral sqrt(p0*p1) dx via importance-sampled
    Monte Carlo against a mixture of the two KDEs (so the sampling
    distribution has support wherever either class does)."""
    rng = np.random.default_rng(seed)
    kde0 = gaussian_kde(features0.T)
    kde1 = gaussian_kde(features1.T)

    n0 = len(features0)
    from_0 = rng.random(n_mc) < 0.5
    samples = np.empty((n_mc, features0.shape[1]))
    samples[from_0] = kde0.resample(from_0.sum(), seed=rng).T
    samples[~from_0] = kde1.resample((~from_0).sum(), seed=rng).T

    p0 = kde0(samples.T)
    p1 = kde1(samples.T)
    mixture_density = 0.5 * p0 + 0.5 * p1
    integrand = np.sqrt(np.clip(p0 * p1, 0, None))
    # importance sampling estimate of integral sqrt(p0*p1) dx using the
    # 50/50 mixture as the proposal distribution
    bc = float(np.mean(integrand / np.clip(mixture_density, 1e-300, None)))
    bound = float(np.sqrt(prior0 * prior1) * bc)
    return {"bhattacharyya_coefficient": bc, "bayes_error_bound": bound, "n_mc": n_mc}


def _gaussian_differential_entropy(cov):
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)


def mutual_information_gaussian_mixture(features0, features1, prior0=0.5, prior1=0.5, n_mc=20000, seed=0):
    """I(Y;X) for a two-component Gaussian mixture fit to the real class
    features, estimated via Monte Carlo entropy of the mixture density
    (H(X|Y) is closed-form; H(X) is not, for a mixture, so it's estimated
    by sampling)."""
    mean0, cov0 = fit_gaussian(features0)
    mean1, cov1 = fit_gaussian(features1)
    rng = np.random.default_rng(seed)

    from_0 = rng.random(n_mc) < prior0
    samples = np.empty((n_mc, features0.shape[1]))
    samples[from_0] = rng.multivariate_normal(mean0, cov0, size=from_0.sum())
    samples[~from_0] = rng.multivariate_normal(mean1, cov1, size=(~from_0).sum())

    from scipy.stats import multivariate_normal
    p0 = multivariate_normal(mean0, cov0, allow_singular=True).pdf(samples)
    p1 = multivariate_normal(mean1, cov1, allow_singular=True).pdf(samples)
    mixture_density = prior0 * p0 + prior1 * p1
    h_x = float(-np.mean(np.log(np.clip(mixture_density, 1e-300, None))))

    h_x_given_y = prior0 * _gaussian_differential_entropy(cov0) + prior1 * _gaussian_differential_entropy(cov1)
    mi = h_x - h_x_given_y
    return {"mutual_information_nats": float(mi), "h_x": h_x, "h_x_given_y": float(h_x_given_y)}


# ---------------------------------------------------------------------------
# Self-tests on synthetic Gaussians with KNOWN separation, so the real-data
# numbers can be trusted once these pass.
# ---------------------------------------------------------------------------


def test_bhattacharyya_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    features0 = rng.normal(0, 1, size=(500, 3))
    features1 = rng.normal(0, 1, size=(500, 3))
    result = bayes_error_bound_gaussian(features0, features1)
    # identical distributions -> Bhattacharyya coefficient near 1, Bayes
    # error bound near the chance rate (0.5 for equal priors)
    assert result["bhattacharyya_coefficient"] > 0.9, result
    assert result["bayes_error_bound"] > 0.4, result


def test_bhattacharyya_bound_shrinks_as_classes_separate():
    rng = np.random.default_rng(0)
    features0 = rng.normal(0, 1, size=(500, 3))
    bounds = []
    for sep in [0.0, 2.0, 5.0, 10.0]:
        features1 = rng.normal(sep, 1, size=(500, 3))
        result = bayes_error_bound_gaussian(features0, features1)
        bounds.append(result["bayes_error_bound"])
    # monotonically non-increasing as the classes separate further
    assert all(bounds[i] >= bounds[i + 1] - 1e-6 for i in range(len(bounds) - 1)), bounds
    assert bounds[-1] < 0.05, bounds  # well-separated classes -> near-zero Bayes error


def test_chernoff_at_half_matches_bhattacharyya():
    rng = np.random.default_rng(3)
    features0 = rng.normal(0, 1, size=(400, 2))
    features1 = rng.normal(2.5, 1, size=(400, 2))
    mean0, cov0 = fit_gaussian(features0)
    mean1, cov1 = fit_gaussian(features1)
    mu_half = chernoff_gaussian(mean0, cov0, mean1, cov1, 0.5)
    b_dist, _ = bhattacharyya_gaussian(mean0, cov0, mean1, cov1)
    # chernoff_gaussian(s=0.5) should equal the Bhattacharyya distance
    assert abs(mu_half - b_dist) < 1e-9, (mu_half, b_dist)


def test_optimized_chernoff_bound_at_least_as_tight_as_bhattacharyya():
    rng = np.random.default_rng(4)
    # asymmetric covariances, where the optimal s is expected to move away
    # from 0.5 and the optimized bound should be strictly tighter
    features0 = rng.normal(0, 1, size=(600, 2))
    features1 = rng.normal(2.0, 3.0, size=(600, 2))
    gauss = bayes_error_bound_gaussian(features0, features1)
    chernoff = optimized_chernoff_bound_gaussian(features0, features1)
    assert chernoff["bayes_error_bound"] <= gauss["bayes_error_bound"] + 1e-9, (gauss, chernoff)


def test_kde_bound_agrees_with_gaussian_bound_on_gaussian_data():
    rng = np.random.default_rng(1)
    features0 = rng.normal(0, 1, size=(400, 2))
    features1 = rng.normal(3, 1, size=(400, 2))
    gauss = bayes_error_bound_gaussian(features0, features1)
    kde = bayes_error_bound_kde(features0, features1, n_mc=4000, seed=1)
    # both estimators should agree to within a reasonable tolerance on data
    # that actually satisfies the Gaussian assumption
    assert abs(gauss["bayes_error_bound"] - kde["bayes_error_bound"]) < 0.05, (gauss, kde)


def test_mutual_information_increases_with_separation():
    rng = np.random.default_rng(2)
    features0 = rng.normal(0, 1, size=(1000, 2))
    mis = []
    for sep in [0.0, 2.0, 5.0]:
        features1 = rng.normal(sep, 1, size=(1000, 2))
        result = mutual_information_gaussian_mixture(features0, features1, seed=2)
        mis.append(result["mutual_information_nats"])
    assert mis[0] < mis[1] < mis[2], mis
    assert mis[0] < 0.05, mis  # no separation -> ~0 mutual information with the label


if __name__ == "__main__":
    test_bhattacharyya_zero_for_identical_distributions()
    test_bhattacharyya_bound_shrinks_as_classes_separate()
    test_chernoff_at_half_matches_bhattacharyya()
    test_optimized_chernoff_bound_at_least_as_tight_as_bhattacharyya()
    test_kde_bound_agrees_with_gaussian_bound_on_gaussian_data()
    test_mutual_information_increases_with_separation()
    print("All Bayes-bound theory self-tests passed.")
