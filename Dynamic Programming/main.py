import argparse
import agents
import environment
import runner
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='test bed for dynamic programming algorithms')

subparsers = parser.add_subparsers(dest='agent')
subparsers.required = True

parser_RD = subparsers.add_parser(
    'RD', description='Random Agent')
parser_VI = subparsers.add_parser(
    'VI', description='Value Iteration agent')
parser_PI = subparsers.add_parser(
    'PI', description='Policy Iteration agent')
parser_QL = subparsers.add_parser(
    'QL', description='Q-Learning agent')
parser_ALL = subparsers.add_parser(
    'ALL', description='Run all agent (if implemnted) and plot performance')

parsers = [parser_RD, parser_VI, parser_PI, parser_QL, parser_ALL]

arg_dico = {'RD': agents.Agent,
            'VI': agents.ValueIteration,
            'PI': agents.PolicyIteration,
            'QL': agents.QLearning
            }

def plot_results(sum_of_rewards, list_legends):
    for sum_rew, legend in zip(sum_of_rewards, list_legends):
        plt.plot(sum_rew, label=legend)
    plt.legend(loc='lower right')
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards')
    plt.show()

def plot_policy(policy, V, name):
    # Action in [UP, RIGHT, DOWN, LEFT]
    action = {0: "T", 1: "R", 2: "D", 3: "L"}

    for id_r, row in enumerate(policy.argmax(1).reshape((4, 12))):
        for id_c, col in enumerate(row):
            if id_r != 3:
                print(action[col], end="\t")
            else:
                if id_c == 0:
                    print(action[col], end="\t")
                elif id_c == 11:
                    print("G", end="\t")
                else:
                    print("-", end="\t")
        print("")

def run_agent(nb_episodes, args):
    env_class = environment.Environment()
    agent_class = arg_dico[args.agent]

    print("Running a single instance simulation...")
    name = args.agent
    my_runner = runner.Runner(env_class, agent_class(env_class), name)
    if name in ["RD", "QL"]:
        final_reward = my_runner.loop(nb_episodes)
        plot_results([final_reward], [args.agent])
    elif name in ["PI", "VI"]:
        policy, V = my_runner.loop(nb_episodes)
        plot_policy(policy, V, name)

def main():
    nb_episodes = 500
    args = parser.parse_args()
    if args.agent != "ALL":
        run_agent(nb_episodes, args)
    else:
        list_final_reward = []
        list_agent = []
        for agent in ["QL", "RD"]:
            env_class = environment.Environment()
            agent_class = arg_dico[agent]

            print("Running a single instance simulation...")
            my_runner = runner.Runner(env_class, agent_class(env_class), agent)
            final_reward = my_runner.loop(nb_episodes)
            list_final_reward.append(final_reward)
            list_agent.append(agent)

        plot_results(list_final_reward, list_agent)

if __name__ == "__main__":
    main()
