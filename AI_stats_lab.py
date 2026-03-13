import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def posterior_probability(time):
    """
    Compute P(B | X = time) using Bayes rule.
    """

    prior_A = 0.3
    prior_B = 0.7

    likelihood_A = np.exp(-((time - 40) ** 2) / 4)
    likelihood_B = np.exp(-((time - 45) ** 2) / 4)

    numerator = prior_B * likelihood_B
    denominator = prior_A * likelihood_A + numerator

    return numerator / denominator

def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    prior_A = 0.3
    prior_B = 0.7

    labels = np.random.choice(['A', 'B'], size=n, p=[prior_A, prior_B])

    samples = np.zeros(n)

    samples[labels == 'A'] = np.random.normal(40, 2, np.sum(labels == 'A'))
    samples[labels == 'B'] = np.random.normal(45, 2, np.sum(labels == 'B'))

    mask = np.abs(samples - time) < 0.1
    filtered_labels = labels[mask]

    if len(filtered_labels) == 0:
        return 0

    return np.sum(filtered_labels == 'B') / len(filtered_labels)