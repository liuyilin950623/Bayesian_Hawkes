from src.utils import *


class Simulation(object):

    def __init__(self, alpha_mu=0.01, beta_mu=0.01, alpha_kappa=0.01, beta_kappa=0.01, alpha_beta=0.01, beta_beta=0.01):
        self.alpha_mu = alpha_mu
        self.beta_mu = beta_mu
        self.alpha_kappa = alpha_kappa
        self.beta_kappa = beta_kappa
        self.alpha_beta = alpha_beta
        self.beta_beta = beta_beta

    def infer_B(self, timestamp, mu, beta, kappa):
        """
        This method infers latent variable Bi, i = 1, . . . , n
            for each event that labels whether it is a background or child event
        This is done via
            1. create a normalised intensity list
            2. sample from a multinomial distribution
        :param timestamp: time stamp of VIX hiking event
        :param mu: background intensity
        :param beta: exponential decay rate
        :param kappa: decay multiplication constant
        :return: latent variable Bi
        """
        intensity = []
        for (i, t_j) in enumerate(timestamp):
            t_i = [mu]
            for j in range(i - 1):
                time_distance = t_j - timestamp[j]
                t_i.append(kappa * beta * np.exp(time_distance * -beta))
            normalised_intensities = [i / sum(t_i) for i in t_i]
            intensity.append(normalised_intensities)

        latent_list = []
        for k in intensity:
            latent = np.random.choice(np.arange(len(k)), p=k)
            latent_list.append(timestamp[latent] if latent > 0 else 0)

        return latent_list

    def _simulate_gamma_distribution(self, shape, scale, number_generated=1):
        return np.random.gamma(shape, scale, number_generated)

    def simulate_mu(self, partition):
        """
        This method simulates mu from Gamma(alpha_mu+|S_0|, beta_mu +1)
        :param partition: inferred latent variable Bi, i = 1, . . . , n
        for each event that labels whether it is a background or child event
        :return: simulated mu
        """
        parent_event_number = len(partition) - np.count_nonzero(partition)
        return self._simulate_gamma_distribution(shape=self.alpha_mu+parent_event_number,
                                                 scale=1/(self.beta_mu+1))

    def simulate_kappa(self, max_T, partition, beta):
        """
        This method simulates kappa from Gamma(Sum(|S_j|)+alpha_kappa, G_tilda+beta_kappa)
        :param max_T: length of the period
        :param partition: inferred latent variable Bi, i = 1, . . . , n
        for each event that labels whether it is a background or child event
        :param beta: exponential decay rate
        :return: simulated kappa
        """
        G = compile_G(max_T, partition, beta)
        return self._simulate_gamma_distribution(shape=np.count_nonzero(partition)+self.alpha_kappa,
                                                 scale=1/(self.beta_kappa+G))

    def simulate_beta(self, time_stamp, partition):
        """
        This method simulates beta from Gamma(N_child+alpha_beta, tau+beta_beta)
        :param time_stamp: time stamp of VIX spikes
        :param partition: inferred latent variable Bi, i = 1, . . . , n
        for each event that labels whether it is a background or child event
        :return: simulated beta
        """
        N_child_events, tau = compile_tau(time_stamp, partition)
        return self._simulate_gamma_distribution(shape=N_child_events+self.alpha_beta,
                                                 scale=1/(tau+self.beta_beta))
