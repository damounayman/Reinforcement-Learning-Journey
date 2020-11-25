import numpy as np
import random
random.seed(0)
np.random.seed(0)


"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        return np.random.randint(0, 10)

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass

class epsGreedyAgent:
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.mu = {a:[] for a in self.A}
        self.epsilon = 0.1

    def choose(self):
        """Acts in the environment.
        returns the chosen action.
        """
        for a in self.A:
            if len(self.mu[a]) == 0:
                return a
        if np.random.uniform(0, 1) < 1 - self.epsilon:
            return np.argmax([np.mean(self.mu[a]) for a in self.A])
        else:
            return np.random.randint(0, 10)

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[action].append(reward)


class BesaAgent():
    # https://hal.archives-ouvertes.fr/hal-01025651v1/document
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.arm_count = np.zeros(len(self.A), dtype=int) # Number of arm pulled.
        self.rewards = [[] for _ in range(len(self.A))] # rewards for each arm.

    def BESA_2(self, action_1, action_2):
        # If one of the subset is empty :
        if (len(action_1) == 0) or (len(action_2) == 0):
            return (action_1 + action_2)[0]
        else:
            # Break action_1 in half
            a = self.BESA_2(action_1[:int(len(action_1)/2)], action_1[int(len(action_1)/2):])
            # Break action_2 in half
            b = self.BESA_2(action_2[:int(len(action_2)/2)], action_2[int(len(action_2)/2):])
            # if N[a] < N[b], select I_a entirely.
            I_a = random.sample(list(range(self.arm_count[a])), int(min(self.arm_count[a],self.arm_count[b])))
            # if N[b] < N[a], select I_a entirely.
            I_b = random.sample(list(range(self.arm_count[b])), int(min(self.arm_count[b],self.arm_count[a])))
            mu_a = np.mean([self.rewards[a][i] for i in I_a])
            mu_b = np.mean([self.rewards[b][i] for i in I_b])
            # Returns the arm with the highest empirical mean on the subsets.
            return (mu_a >= mu_b)*a+(mu_a < mu_b)*b


    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        if 0 in list(self.arm_count): # No arms hasn't been pulled
            # Pull one of them
            a = list(self.arm_count).index(0)
        else:
            # all the arms have been pulled
            a = self.BESA_2(self.A[:int(len(self.A)/2)], self.A[int(len(self.A)/2):])
        return a


    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.
        This is where your agent can learn.
        """
        self.arm_count[action] += 1
        self.rewards[action] += [reward]

class SoftmaxAgent:
    # https://www.cs.mcgill.ca/~vkules/bandits.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.arm_count = np.zeros(len(self.A))
        self.mu = np.zeros(len(self.A))
        self.tau = 0.2

    def choose(self):
        exp = np.exp((self.mu - np.max(self.mu)) / self.tau)
        boltzmann_dis = exp/np.sum(exp)
        a = np.random.choice(self.A, p=boltzmann_dis)
        return a


    def update(self, action, reward):
        self.arm_count[action] += 1
        self.mu[action] += (reward - self.mu[action]) / self.arm_count[action]



class UCBAgent:
    # https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    def __init__(self):
        self.A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.arm_count = np.zeros(len(self.A))
        self.mu = np.zeros(len(self.A))
        self.n = 0
    def choose(self):
        """Acts in the environment.
        returns the chosen action.
        """
        self.n += 1
        if self.n < (len(self.A) + 1):
            return self.n - 1

        empirical_averages = self.mu / self.arm_count
        UCB = np.sqrt(2 * np.log(self.n) / self.arm_count)
        bounds = empirical_averages + UCB
        optimal_policy = np.argmax(bounds)
        return optimal_policy

    def update(self, a, r):
        self.arm_count[a] += 1
        self.mu[a] += r

class ThompsonAgent:
    # https://en.wikipedia.org/wiki/Thompson_sampling
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.successes = np.ones(len(self.A))*100
        self.fails = np.ones(len(self.A))*1000


    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        # Compute probability of theta with beta distribution
        theta = np.random.beta(a=(self.successes + 1), b=(self.fails + 1))

        return np.argmax(theta)

    def update(self, a, r): # Adjust distribution
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        # updates the relevant values for the chosen arm
        # total counts of rewards of arm
        self.successes[a] += r
        # total counts of failed rewards on arm
        self.fails[a] += 1 - r

class KLUCBAgent:
    # See: https://hal.archives-ouvertes.fr/hal-00738209v2
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.K = len(self.A)
        self.N = np.zeros(self.K) #Number of times the arm was pulled
        self.S = np.zeros(self.K)
        self.precision = 1e-6
        self.max_iterations = 100

    # compute Kullback-Leibler divergence
    def kl(self, x, y):
        eps = 1e-15
        x = min(max(x, eps), 1 - eps)
        y = min(max(y, eps), 1 - eps)
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

    def klucb_upper(self,kl,N, S, k, t, precision=1e-6, max_iterations=50):
        """
        Compute the upper confidence bound for each arm
        """
        upperbound = np.log(t) / N[k]
        reward = S[k] / N[k]
        u = upperbound
        l = reward
        n = 0
        while n < max_iterations and u - l > precision:
            q = (l + u) / 2
            if kl(reward, q) > upperbound:
                u = q
            else:
                l = q
            n += 1
        return (l + u) / 2

    def choose(self):
        """Acts in the environment.
        returns the chosen action.
        """
        t = np.sum(self.N)
        indices = np.zeros(self.K)
        for k in range(self.K):
            if self.N[k] == 0:
                return k
            # KL-UCB index
            indices[k] = self.klucb_upper(self.kl, self.N, self.S, k, t, self.precision, self.max_iterations)
        selected_arm = np.argmax(indices)
        return selected_arm

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.
        This is where your agent can learn.
        """
        self.N[a] += 1
        self.S[a] += r
