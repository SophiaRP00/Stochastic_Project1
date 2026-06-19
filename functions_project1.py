import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import pandas as pd

## GENERAL FUNCTIONS
def conf_interval(data, confidence_level=0.95):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin_of_error = z * (std_dev / np.sqrt(len(data)))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

def conf_interval_std(data, confidence_level=0.95):
    std_dev = np.std(data, ddof=1)
    chi2_lower = stats.chi2.ppf((1 - confidence_level) / 2, df=len(data) - 1)
    chi2_upper = stats.chi2.ppf(1 - (1 - confidence_level) / 2, df=len(data) - 1)
    lower_bound = std_dev * np.sqrt((len(data) - 1) / chi2_upper)
    upper_bound = std_dev * np.sqrt((len(data) - 1) / chi2_lower)
    return lower_bound, upper_bound

def chi_square_test(observed, expected):
    chi2_statistic = np.sum((observed - expected) ** 2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_statistic, df=len(observed) - 1)
    return chi2_statistic, p_value

def control_variate(Y, X, c):
    mu_X = np.mean(X)
    return Y - c * (X - mu_X)

### PART I
Probability_matrix = np.array([[0.9915, 0.005, 0.0025,0, 0.001],[0, 0.986, 0.005, 0.004, 0.005],[0,0,0.992,0.003,0.005],[0,0,0,0.991,0.009],[0,0,0,0,1]])

def cancer_simulation_women(n):
    results = []

    for _ in range(n):
        state = 0
        time = 0
        history = [0]

        while state != 4:
            state = np.random.choice(5, p=Probability_matrix[state])
            time += 1
            history.append(state)

        results.append((time, history))

    return results

def proportion_local_recurrence(n):
    count = 0

    for _ in range(n):
        state = 0
        visited_local = False

        while state != 4:
            state = np.random.choice(5, p=Probability_matrix[state])

            if state == 1:
                visited_local = True

        if visited_local:
            count += 1

    return count / n

def state_distribution_at_time_t(n, t):
    state_counts = np.zeros(5)

    for _ in range(n):
        state = 0
        for _ in range(t):
            state = np.random.choice(5, p=Probability_matrix[state])
        state_counts[state] += 1

    return state_counts / n

def cancer_simulation_recurrence(n):
    accepted_times = []

    while len(accepted_times) < n:
        death_time, history = cancer_simulation_women(1)[0]
        survived_12_months = death_time >= 12
        recurrence = any(state in [1,2] for state in history[:13])  # Check for recurrence in the first 12 months
        
        if survived_12_months and recurrence:
            accepted_times.append(death_time)
    
    return accepted_times

            break

        Q_est = Q_new

    return Q_est
