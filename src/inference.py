from src.preprocessing import *
from src.simulation import *

import argparse

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '--initial_beta', default=[0.1],
    help='prior estimate for beta'
)
parser.add_argument(
    '--initial_mu', default=[0.1],
    help='prior estimate for mu'
)
parser.add_argument(
    '--initial_kappa', default=[0.1],
    help='prior estimate for kappa'
)
parser.add_argument(
    '--num_iteration', default=100,
    help='number of iteration'
)


def main(beta, mu, kappa, num_iteration):
    """
    Main function that performs inference via Gibbs sampling
    :param beta: initial list of beta with the initial guess as the item
    :param mu: initial list of mu with the initial guess as the item
    :param kappa: initial list of kappa with the initial guess as the item
    :param num_iteration: number of iterations until convergence
    :return: list_of_beta, list_of_mu, list_of_kappa, latent_list to distinguish parent vs child, time_stamp of events
    """
    time_stamp, max_T = get_time_stamp()
    print(time_stamp, max_T)
    simulate = Simulation()

    for i in range(num_iteration):
        latent_list = simulate.infer_B(time_stamp, mu[-1], beta[-1], kappa[-1])
        print(latent_list)

        mu_simulated = simulate.simulate_mu(latent_list)[0] / max_T
        print(mu_simulated)
        mu.append(mu_simulated)

        kappa_simulated = simulate.simulate_kappa(max_T, latent_list, beta[-1])[0]
        print(kappa_simulated)
        kappa.append(kappa_simulated)

        beta_simulated = simulate.simulate_beta(time_stamp, latent_list)[0]
        print(beta_simulated)
        beta.append(beta_simulated)

    return beta, mu, kappa, latent_list, time_stamp


if __name__ == '__main__':
    args = parser.parse_args()
    beta, mu, kappa, latent_list, time_stamp = main(args.initial_beta,
                                                    args.initial_mu,
                                                    args.initial_kappa,
                                                    args.num_iteration)
    plot(mu, kappa, beta)
    print("Simulation Completed. beta, mu, kappa, latent_list and time_stamp are available")
