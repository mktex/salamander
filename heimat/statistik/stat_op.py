import numpy as np
import scipy.stats

# Funnel Beispiel
# udacityp3.anzahl_zeiteinheiten(3250, 520, 570, 0.025, 0.2)
anzahl_zeiteinheiten = lambda N1, N2, N2_star, alpha, beta: \
    experiment_size(p_null=N2/N1, p_alt=N2_star/N1, alpha=(alpha/2.0), beta=.20) / N1 * 2


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def two_proportions_test(success_a, size_a, success_b, size_b):
    """
    # Quelle: http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html
    A/B test for two proportions
    given a success a trial size of group A and B compute
    its zscore and pvalue

    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    Returns
    -------
    zscore : float
        test statistic for the two proportion z-test

    pvalue : float
        p-value for the two proportion z-test

    ACHTUNG: das hier ist gleich statsmodels.api.stats.proportions_ztest
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    prop_pooled = (success_a + success_b) / (size_a + size_b)
    var = prop_pooled * (1 - prop_pooled) * (1 / size_a + 1 / size_b)
    zscore = np.abs(prop_b - prop_a) / np.sqrt(var)
    one_side = 1 - scipy.stats.norm(loc=0, scale=1).cdf(zscore)
    pvalue = one_side * 2
    return zscore, pvalue


def two_proportions_confint(success_a, size_a, success_b, size_b, significance=0.05):
    """
    Quelle: http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html

    A/B test for two proportions
    given a success a trial size of group A and B compute
    its confidence interval
    resulting confidence interval matches R's prop.test function

    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    significance : float, default 0.05
        Often denoted as alpha. Governs the chance of a false positive.
        A significance level of 0.05 means that there is a 5% chance of
        a false positive. In other words, our confidence level is
        1 - 0.05 = 0.95

    Returns
    -------
    prop_diff : float
        Difference between the two proportion

    confint : 1d ndarray
        Confidence interval of the two proportion test
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    var = prop_a * (1 - prop_a) / size_a + prop_b * (1 - prop_b) / size_b
    se = np.sqrt(var)
    # z critical value
    confidence = 1 - significance
    z = scipy.stats.norm(loc=0, scale=1).ppf(confidence + significance / 2)
    # standard formula for the confidence interval
    # point-estimtate +- z * standard-error
    prop_diff = prop_b - prop_a
    confint = prop_diff + np.array([-1, 1]) * z * se
    return prop_diff, confint


def sample_power_probtest(p1, p2, power=0.8, sig=0.05, alternative="one-sided"):
    """
        https://stackoverflow.com/questions/15204070/is-there-a-python-scipy-function-to-determine-parameters-needed-to-obtain-a-ta
    """
    from scipy.stats import norm
    if alternative == "one-sided":
        z = norm.isf([sig])
    else:
        z = norm.isf([sig / 2])  # two-sided t test
    zp = -1 * norm.isf([power])
    d = (p1 - p2)
    s = 2 * ((p1 + p2) / 2) * (1 - ((p1 + p2) / 2))
    n = s * ((zp + z) ** 2) / (d ** 2)
    return int(np.round(n[0]))


def sample_power_difftest(d, s, power=0.8, sig=0.05, alternative="one-sided"):
    """
    :param d: Differenz der Durchschnitte
    :param s: Standardabweichung
    :return:
    """
    from scipy.stats import norm
    if alternative == "one-sided":
        z = norm.isf([sig])
    else:
        z = norm.isf([sig / 2])
    zp = -1 * norm.isf([power])
    n = (2 * (s ** 2)) * ((zp + z) ** 2) / (d ** 2)
    return int(np.round(n[0]))


def plot_power(powerObj=None, dep_var='nobs', nobs=None, effect_size=None,
               alpha=0.05, ax=None, title=None, precision=2, plt_kwds=None, **kwds):
    """
        # aus dem Quellcode statsmodels
        # wegen benötigte Anpassungen übernommen
    """
    from statsmodels.graphics import utils
    from statsmodels.graphics.plottools import rainbow
    fig, ax = utils.create_mpl_ax(ax)
    import matplotlib.pyplot as plt
    colormap = plt.cm.Paired
    plt_alpha = 1
    lw = 2
    if dep_var == 'nobs':
        colors = rainbow(len(effect_size))
        colors = [colormap(i) for i in np.linspace(0, 1.0, len(effect_size))]
        for ii, es in enumerate(effect_size):
            power = powerObj.power(es, nobs, alpha, **kwds)
            ax.plot(nobs, power, lw=lw, alpha=plt_alpha,
                    color=colors[ii], label=('es=%.' + str(precision) + 'f') % es)
            xlabel = 'Number of Observations'
    elif dep_var in ['effect size', 'effect_size', 'es']:
        colors = rainbow(len(nobs))
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(nobs))]
        for ii, n in enumerate(nobs):
            power = powerObj.power(effect_size, n, alpha, **kwds)
            ax.plot(effect_size, power, lw=lw, alpha=plt_alpha,
                    color=colors[ii], label=('N=%.' + str(precision) + 'f') % n)
            xlabel = 'Effect Size'
    elif dep_var in ['alpha']:
        # experimental nobs as defining separate lines
        colors = rainbow(len(nobs))
        for ii, n in enumerate(nobs):
            power = powerObj.power(effect_size, n, alpha, **kwds)
            ax.plot(alpha, power, lw=lw, alpha=plt_alpha,
                    color=colors[ii], label=('N=%.' + str(precision) + 'f') % n)
            xlabel = 'alpha'
    else:
        raise ValueError('depvar not implemented')
    if title is None:
        title = 'Power of Test'
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig


def do_bayesian_update(self, p, r, m, doPlot=False, return_trace=False, nsamp=101):
    """
        Implementierung der Grid-Methode für die Bayes-Aktualisierung einer Beta-Prior
        P(theta | D, M) = [P(D | theta, M) * P(theta | M)] / P(D | M)
        Posterior: P(theta | D, M)
        Likelyhood: P(D | theta, M)
        Prior: P(theta | M)
        Evidence: P(D | M)
        # Für die Käufe wird die Likelyhood mit einer Binomial-Verteilung modeliert.
        #   -> Die Daten werden als r Käufe (Produkt A & Produkt B) aus aus m (Produkt A) Käufe analysiert.
        # Die Wahrscheinlichkeit p wird dafür verwendet um die Parameter für die Prior-Dichtefunktion
        # zu berechen. So resultiert eine Prior Binomial(n, k), so dass p = k / n. n wird hier als 100. angenommen
        #   -> oder auch k Kaufe aus n insgesamt
        # Der Resultat ist eine Poster-Dichtefunktion die der Form einer Beta-Funktion annimmt
        #   -> Beta(alpha_post, beta_post), mit alpha_post = alpha_prior + r und beta_post = beta_prior + m - r
    :param p:
    :param m:
    :param r:
    :return:
    """
    import scipy.stats
    import matplotlib.pyplot as plt
    Nsamp = nsamp
    delta = 1. / (Nsamp - 1)
    theta_grid = np.arange(0, 1, delta)
    n = 100
    k = int(p * n)
    alpha_prior = k + 1  # die Beta-Funktion ist equivalent der Binomial-Verteilung mit alpha - 1 = k
    beta_prior = n - k + 1  # beta - 1 = n - k
    # Prior: Beta(alpha_prior, beta_prior)
    prior_dist = scipy.stats.beta(alpha_prior, beta_prior)
    prior_dist_pdf = [prior_dist.pdf(theta_grid[ik]) for ik in range(len(theta_grid))]
    # Likelyhood: Binom(r, m) oder auch r Käufe in m Bestellungen
    likelyhood_pmf = [scipy.stats.binom.pmf(r, m, theta_grid[ik]) for ik in range(len(theta_grid))]
    xsum_like = np.sum(likelyhood_pmf)
    likelyhood_pmf = [x / (delta * xsum_like) for x in likelyhood_pmf]
    # Posterior: Beta(alpha_post, beta_post)
    posterior_dist = scipy.stats.beta(alpha_prior + r, beta_prior + m - r)
    posterior_dist_pdf = [posterior_dist.pdf(theta_grid[ik]) for ik in range(len(theta_grid))]
    # Berechnung der Durchschnitt basiert auf Posterior-Dichtefunktion
    xpostsum = np.sum(posterior_dist_pdf)
    xmean = np.sum(
        [x[0] * x[1] for x in zip(theta_grid, posterior_dist_pdf)]
    ) / xpostsum
    if doPlot:
        plt.scatter(theta_grid, likelyhood_pmf, color="red", s=5)
        plt.scatter(theta_grid, posterior_dist_pdf, color="blue", s=5)
        plt.scatter(theta_grid, prior_dist_pdf, color="cyan", s=5)
        plt.title("Likelyhood (Rot); Prior (Zyan); Posterior (Blau)")
        plt.show()
    if return_trace:
        return xmean, theta_grid, posterior_dist_pdf, prior_dist_pdf
    return xmean


def power(p_null, p_alt, n, alpha=.05, plot=True):
    """
    *** Quelle: Udacity ***
    Compute the power of detecting the difference in two populations with
    different proportion parameters, given a desired alpha rate.

    Input parameters:
        p_null: base success rate under null hypothesis
        p_alt : desired success rate to be detected, must be larger than
                p_null
        n     : number of observations made in each group
        alpha : Type-I error rate
        plot  : boolean for whether or not a plot of distributions will be
                created

    Output value:
        power : Power to detect the desired difference, under the null.
    """
    import matplotlib.pyplot as plt
    # Compute the power
    se_null = np.sqrt((p_null * (1 - p_null) + p_null * (1 - p_null)) / n)
    null_dist = scipy.stats.norm(loc=0, scale=se_null)
    p_crit = null_dist.ppf(1 - alpha)

    se_alt = np.sqrt((p_null * (1 - p_null) + p_alt * (1 - p_alt)) / n)
    alt_dist = scipy.stats.norm(loc=p_alt - p_null, scale=se_alt)
    beta = alt_dist.cdf(p_crit)

    if plot:
        # Compute distribution heights
        low_bound = null_dist.ppf(.01)
        high_bound = alt_dist.ppf(.99)
        x = np.linspace(low_bound, high_bound, 201)
        y_null = null_dist.pdf(x)
        y_alt = alt_dist.pdf(x)

        # Plot the distributions
        plt.plot(x, y_null)
        plt.plot(x, y_alt)
        plt.vlines(p_crit, 0, np.amax([null_dist.pdf(p_crit), alt_dist.pdf(p_crit)]),
                   linestyles='--')
        plt.fill_between(x, y_null, 0, where=(x >= p_crit), alpha=.5)
        plt.fill_between(x, y_alt, 0, where=(x <= p_crit), alpha=.5)

        plt.legend(['null', 'alt'])
        plt.xlabel('difference')
        plt.ylabel('density')
        plt.show()

    # return power
    return (1 - beta)


def experiment_size(p_null, p_alt, alpha=.05, beta=.20):
    """
    *** Quelle: Udacity ***
    Beispiel:
        Funnel: N1 -> N2 -> N3;
        Anzahl zeiteinheiten:
            udacityp3.experiment_size(p_null=N2/N1, p_alt=N2_Neu/N1, alpha=(0.025/2.0), beta=.20) / N1 * 2

    Compute the minimum number of samples needed to achieve a desired power
    level for a given effect size.

    Input parameters:
        p_null: base success rate under null hypothesis
        p_alt : desired success rate to be detected
        alpha : Type-I error rate
        beta  : Type-II error rate

    Output value:
        n : Number of samples required for each group to obtain desired power
    """

    # Get necessary z-scores and standard deviations (@ 1 obs per group)
    z_null = scipy.stats.norm.ppf(1 - alpha)
    z_alt = scipy.stats.norm.ppf(beta)
    sd_null = np.sqrt(p_null * (1 - p_null) + p_null * (1 - p_null))
    sd_alt = np.sqrt(p_null * (1 - p_null) + p_alt * (1 - p_alt))

    # Compute and return minimum sample size
    p_diff = p_alt - p_null
    n = ((z_null * sd_null - z_alt * sd_alt) / p_diff) ** 2
    return np.ceil(n)


def quantile_ci(data, q, c=.95, n_trials=1000):
    """
    *** Quelle: Udacity ***
    Compute a confidence interval for a quantile of a dataset using a bootstrap
    method.

    Input parameters:
        data: data in form of 1-D array-like (e.g. numpy array or Pandas series)
        q: quantile to be estimated, must be between 0 and 1
        c: confidence interval width
        n_trials: number of bootstrap samples to perform

    Output value:
        ci: Tuple indicating lower and upper bounds of bootstrapped
            confidence interval

    quantile_permtest(data['time'], data['condition'], 0.9, alternative = 'less')
    """

    # initialize storage of bootstrapped sample quantiles
    n_points = data.shape[0]
    sample_qs = []

    # For each trial...
    for _ in range(n_trials):
        # draw a random sample from the data with replacement...
        sample = np.random.choice(data, n_points, replace=True)

        # compute the desired quantile...
        sample_q = np.percentile(sample, 100 * q)

        # and add the value to the list of sampled quantiles
        sample_qs.append(sample_q)

    # Compute the confidence interval bounds
    lower_limit = np.percentile(sample_qs, (1 - c) / 2 * 100)
    upper_limit = np.percentile(sample_qs, (1 + c) / 2 * 100)

    return (lower_limit, upper_limit)


def quantile_permtest(x, y, q, alternative='less', n_trials=10000):
    """
    *** Quelle: Udacity ***
    Compute a confidence interval for a quantile of a dataset using a bootstrap
    method.

    Input parameters:
        x: 1-D array-like of data for independent / grouping feature as 0s and 1s
        y: 1-D array-like of data for dependent / output feature
        q: quantile to be estimated, must be between 0 and 1
        alternative: type of test to perform, {'less', 'greater'}
        n_trials: number of permutation trials to perform

    Output value:
        p: estimated p-value of test
    """

    # initialize storage of bootstrapped sample quantiles
    sample_diffs = []

    # For each trial...
    for _ in range(n_trials):
        # randomly permute the grouping labels
        labels = np.random.permutation(y)

        # compute the difference in quantiles
        cond_q = np.percentile(x[labels == 0], 100 * q)
        exp_q = np.percentile(x[labels == 1], 100 * q)

        # and add the value to the list of sampled differences
        sample_diffs.append(exp_q - cond_q)

    # compute observed statistic
    cond_q = np.percentile(x[y == 0], 100 * q)
    exp_q = np.percentile(x[y == 1], 100 * q)
    obs_diff = exp_q - cond_q

    # compute a p-value
    if alternative == 'less':
        hits = (sample_diffs <= obs_diff).sum()
    elif alternative == 'greater':
        hits = (sample_diffs >= obs_diff).sum()

    return (hits / n_trials)


def ranked_sum(x, y, alternative='two-sided'):
    """
    *** Quelle: Udacity ***
    Return a p-value for a ranked-sum test, assuming no ties.

    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}

    Output value:
        p: estimated p-value of test
    """

    # compute U
    u = 0
    for i in x:
        wins = (i > y).sum()
        ties = (i == y).sum()
        u += wins + 0.5 * ties

    # compute a z-score
    n_1 = x.shape[0]
    n_2 = y.shape[0]
    mean_u = n_1 * n_2 / 2
    sd_u = np.sqrt(n_1 * n_2 * (n_1 + n_2 + 1) / 12)
    z = (u - mean_u) / sd_u

    # compute a p-value
    if alternative == 'two-sided':
        p = 2 * scipy.stats.norm.cdf(-np.abs(z))
    if alternative == 'less':
        p = scipy.stats.norm.cdf(z)
    elif alternative == 'greater':
        p = scipy.stats.norm.cdf(-z)

    return p


def sign_test(x, y, alternative='two-sided'):
    """
    *** Quelle: Udacity ***
    Return a p-value for a ranked-sum test, assuming no ties.
    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}

    Output value:
        p: estimated p-value of test
    """

    # compute parameters
    n = x.shape[0] - (x == y).sum()
    k = (x > y).sum() - (x == y).sum()

    # compute a p-value
    if alternative == 'two-sided':
        p = min(1, 2 * scipy.stats.binom(n, 0.5).cdf(min(k, n - k)))
    if alternative == 'less':
        p = scipy.stats.binom(n, 0.5).cdf(k)
    elif alternative == 'greater':
        p = scipy.stats.binom(n, 0.5).cdf(n - k)

    return p


def get_confusion_matrix_stats(cm, i):
    """
        Given a Confusion Matrix cm, calculates precision, recall and F1 scores
    :param cm: confusion matrix
    :param i: position of the variable, for with the caculation be done
    :return: three statistics: precision, recall and the F1-Score
    """
    tp = cm[i, i]
    fp = np.sum(cm[i, :]) - tp
    fn = np.sum(cm[:, i]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score
