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

#### PART II
Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],[0, -0.014, 0.005, 0.004, 0.005],[0, 0, -0.008, 0.003, 0.005],[0, 0, 0, -0.009, 0.009],[0, 0, 0, 0, 0]])
def continuous_cancer_simulation(n, Q):
    results = []

    for _ in range(n):
        state = 0
        time = 0
        history = [0]
        trajectory = [(0.0, state)]

        while state != 4:
            rates = Q[state]
            total_rate = -rates[state]
            
            time += np.random.exponential(1 / total_rate)
            probabilities = np.where(rates > 0, rates / total_rate, 0)
            state = np.random.choice(5, p=probabilities)
            history.append(state)
            trajectory.append((time, state))

        results.append((time, history, trajectory))

    return results

def continuous_recurrence(n, time_threshold, state_threshold):
    count = 0
    for _ in range(n):
        state = 0
        time = 0

        while state != 4:
            rates = Q[state]
            total_rate = -rates[state]
            waiting_time = np.random.exponential(1 / total_rate)
            if (time + waiting_time) >= time_threshold:
                break

            time += waiting_time

            probabilities = np.where(rates > 0, rates / total_rate, 0)
            state = np.random.choice(5, p=probabilities)
            
        if state == state_threshold:
            count += 1

    return count / n

def kaplan_meier_estimate(data):
    times = np.array([record[0] for record in data])
    unique_times, deaths = np.unique(times, return_counts=True)
    n = len(times)
    survival_prob = 1.0
    x = []
    y = []
    for t, d in zip(unique_times, deaths):
        x.append(t)
        y.append(survival_prob)
        survival_prob *= (n - d) / n
        n -= d

    return np.array(x), np.array(y)
    

### Part III
def time_series_observation(jump_times, states, observation_gap):

    observations = []
    t_obs = 0

    while True:

        idx = np.searchsorted(jump_times, t_obs, side="right") - 1
        current_state = states[idx]

        observations.append(current_state)

        if current_state == 4: 
            break

        t_obs += observation_gap

    return observations


def CTMC_simulator(Q_est, start_state, end_state, T=48):
    n_states = Q_est.shape[0]

    while True:
        t = 0.0
        state = start_state

        N = np.zeros((n_states, n_states), dtype=float)
        S = np.zeros(n_states, dtype=float)

        while t < T:
            rate = -Q_est[state, state]
            if rate <= 0:
                break

            wait_time = np.random.exponential(1 / rate)

            if t + wait_time > T:
                S[state] += T - t
                break

            S[state] += wait_time               
            t += wait_time

            probs = Q_est[state].copy()
            probs[state] = 0
            probs = probs / probs.sum()

            new_state = np.random.choice(n_states, p=probs)

            N[state, new_state] += 1
            state = new_state

        if state == end_state:
            return N, S
        
def MCEM(observed_series, Q0, n_states=5, T=48, tol=1e-3, max_iter=50):
    Q_est = Q0.copy()

    for _ in range(max_iter):

        N_tot = np.zeros((n_states, n_states))
        S_tot = np.zeros(n_states)

        for series in observed_series:

            for t in range(len(series) - 1):
                start = series[t]
                end = series[t + 1]

                N, S = CTMC_simulator(Q_est, start, end, T)

                N_tot += N
                S_tot += S

        Q_new = np.zeros_like(Q_est)

        for i in range(n_states):
            if S_tot[i] > 0:
                Q_new[i, :] = N_tot[i, :] / S_tot[i]

        np.fill_diagonal(Q_new, -Q_new.sum(axis=1))

        # convergence check
        if np.max(np.abs(Q_new - Q_est)) < tol:
            break

        Q_est = Q_new

    return Q_est