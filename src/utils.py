import numpy as np
import matplotlib.pyplot as plt


def kernel(beta, time_interval):
    """ Exponential kernel"""
    return beta * np.exp(-beta * time_interval)


def integrated_kernel(beta, time_interval):
    """ Integrating kernel from 0 to T"""
    return 1-np.exp(-beta * time_interval)


def compile_G(max_T, paritition, beta):
    """ G_tilda is the summation of G(T - t_j) = integral(kernel) for all j
    In practise G_tilda ≈ n as G(T − t_j ) = 0 for the events that are far away from
    the boundary."""
    parent_list = list(set(paritition))[1:]
    # return sum([integrated_kernel(beta, max_T - parent) for parent in parent_list if parent != 0])
    return len(parent_list)


def compile_tau(time_stamp, partition):
    """ tau_(i-j) is the time of a child relative to its parent for all child events"""
    N_child_events = 0
    tau = 0
    for i in range(len(partition)):
        if partition[i] != 0:
            # if it's a child event
            N_child_events += 1
            # take the time difference between the child event and parent event
            tau += (time_stamp[i] - partition[i])
    return N_child_events, tau


def plot(mu, kappa, beta):
    plt.figure(figsize=(12, 6))
    plt.title('Simulated values vs Iteration')
    plt.plot(mu, label='mu')
    plt.plot(kappa, label='kappa')
    plt.plot(beta, label='beta')
    plt.xlabel('# simulations')
    plt.ylabel('simulated values')
    plt.show()
    plt.legend()
