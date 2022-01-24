"""
Course: ECE 517 - Reinforcement Learning
Title: Project 1 - Dynamic Programming - Value Iteration
Author: Christoph Metzner
Date: 09/22/2021
Description: This program simulates a Markov Decision Process (MDP) using dynamic programming (DP). The environment of
the MDP is a class room with an active lecture and the agent is a student arriving late. The goal of the student is to
find the policy that allows him to maximize his learning outcome under consideration of potentially getting sick with
COVID-19. The student want to seat as far up in the front as possible.

The MDP = (S, A, P, R)

States - S:
    - Number of rows (1,..., N)
    - Condition of neighbor seat (0, 1, 2)
        - Seat is empty
        - Seat is taken with masked student
        - Seat is taken with unmasked student

    - One terminal state
Total number of seats: N*3 + 1

The data structure for the states is a 2D-matrix.
The terminal state is represented as one variable with a state-value of 0.

Actions - A: (0, 1)
    - Take seat in current row - stay: 0
    - Go to next row - next: 1
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



# States:
# Number of rows (1,..., N)
# Condition of neighbor seat
#   Seat is empty
#   Seat is taken with masked student
#   Seat is taken with unmasked student
# One terminal state
# Total number of seats: N*3

def get_state_value(V, Q, P, R, row, neighbor_cond, gamma=1, policy_improvement=False):
    """
    Function to compute the state-value and action-values for each state respectively. This function is called during
    a loop by function value_iteration.

    :param V: array of shape (N, 3) containing current state-values of the MDP
    :param Q: array of shape (N, 3, 2) containing current action-values of the MDP
    :param P: Probabilities given by the user
    :param R: Rewards given by the user
    :param row: Current row #
    :param neighbor_cond: Current neighbor condition
    :param gamma: discount factor
    :return: State-value and action-values of current state
    """

    N = Q.shape[0]  # Get row number of the MDP

    # Indices
    # Actions
    A = 2  # two actions
    stay = 0
    next = 1
    # Neighbor seat conditions
    empty = 0
    taken_masked = 1
    taken_nonmasked = 2
    # learn or sick
    sick = 0
    notsick = 1

    # MDP Dynamics / Transition Probabilites:
    # Action: Take seat - 0
    p_sick_empty = P[2]
    p_notsick_empty = 1 - p_sick_empty

    p_sick_masked = P[3]
    p_notsick_masked = 1 - p_sick_masked

    p_sick_nonmasked = P[4]
    p_notsick_nonmasked = 1 - p_sick_nonmasked

    # Action: Go to next row - 1
    p_taken = P[0]
    p_mask = P[1]

    p_empty = 1 - p_taken  # Neighbor seat is empty
    p_taken_masked = p_taken * p_mask  # Neighbor seat is taken with student wearing a mask
    p_taken_nonmasked = p_taken * (1 - p_mask)  # Neighbor seat is taken with student not wearing a mask

    # Data structure [[Action: stay],[Action: next]]
    # Transition Probablitiy Matrix: P
    P = np.array([[[p_sick_empty, p_notsick_empty],  # Action: Stay
                   [p_sick_masked, p_notsick_masked],
                   [p_sick_nonmasked, p_notsick_nonmasked]],
                  [p_empty, p_taken_masked, p_taken_nonmasked]],  # Action: Next
                 dtype=object)

    # Rewards for both actions
    # Rewards action: stay
    R_notsick = 0
    R_sick = R[stay][sick]  # -100
    R_learn = R[stay][1]  # Expression which must be evaluated


    # Compute reward for Learning: R_learn
    # Array with index 0 represents row N --> However, determining R_learn requires to use the actual value of row
    # row index 0 --> actual row N; row index N-1 --> actual row 1
    R_learn = learning_benefit(N=N, row=row, R_learn=R_learn)

    # Rewards action: next
    R_next = R[next][0]  # 0.0
    R_exit = R[next][1]  # -100

    #####################################################
    ### Computation of state-values and action-values ###
    #####################################################

    # Q-value for action: stay
    # The state value of a terminal state has 0
    V_terminal = 0

    # There is a probability of 1 of the student hopefully do some learning
    Q[row, neighbor_cond, stay] = 1 * (R_learn + gamma * V_terminal)\
                                  + (P[stay, neighbor_cond][sick] * (R_sick + gamma * V_terminal)
                                     + P[stay, neighbor_cond][notsick] * (R_notsick + gamma * V_terminal))

    # Check if last row reached, the reward of choosing action "next" changes from 0 to -100 since the student leaves
    # the classroom
    # Q.shape[0] provides N and -1 adjusts for python array starting at 0
    if row == N - 1:  # set reward and state-value for first row. this state-value is terminal --> 0
        # Q-value for action: next
        # Since the action next causes the student to leave the class with probability of 1
        # The student will receive a reward of -100 with 100% certainty
        Q[row, neighbor_cond, next] = 1 * (R_exit + gamma * V_terminal)

    else:  # set reward and state-value for all rows but the first one
        # Q-value for action: next
        #print(Q[row, neighbor_cond, next])
        Q[row, neighbor_cond, next] = P[next, empty] * (R_next + gamma * V[row + 1, empty]) \
                                      + P[next, taken_masked] * (R_next + gamma * V[row + 1, taken_masked]) \
                                      + P[next, taken_nonmasked] * (R_next + gamma * V[row + 1, taken_nonmasked])

    # Select max action-value Q as the next state-value V
    #print(f'Stay: {Q[row, neighbor_cond, stay]} Next: {Q[row, neighbor_cond, next]}')

    # Check if I do policy improvement or not.
    # If False, then I will update the state-value function
    # If True, then the state-value function is not updated anymore only the new action-value function Q
    if policy_improvement is False:
        V[row, neighbor_cond] = max(Q[row, neighbor_cond, stay], Q[row, neighbor_cond, next])

    return V[row, neighbor_cond], [Q[row, neighbor_cond, stay], Q[row, neighbor_cond, next]]


# Function to calculate the learning benefit for each respective learn
def learning_benefit(N, row, R_learn):
    actual_row = N-row # have to convert matrix_index to actual row number in physical classroom
    learning_coefficient = (N - actual_row)  # distance from student (location in classroom) to professor
    row_learning = R_learn * learning_coefficient
    return row_learning


def value_iteration(V, Q, P, R, gamma=1):
    """
    This function performs value iteration to find the optimal state-value function.

    :param V: array of shape (N, 3) containing current state-values of the MDP
    :param Q: array of shape (N, 3, 2) containing current action-values of the MDP
    :param P_user: Probabilities given by the user
    :param R: Rewards given by the user
    :param gamma: discount factor
    :return: multidimensional arrays with state-values and action-values
    """

    MAX_N_ITERS = 120  # max number of iterations
    iterCnt = 0  # iteration counter
    threshold = 0.001  # threshold to measure convergence
    delta = 1  # current maximum difference

    # while loop until convergence (smaller than threshold)
    while ((delta > threshold) and (iterCnt <= MAX_N_ITERS)):
        delta = 0
        oldV = np.copy(V)

        # loop through states
        for row in range(V.shape[0]):
            for neighbor_cond in range(V.shape[1]):
                #print("State")
                #print(f"row: {row+1} and neighbor: {neighbor_cond}")
                V[row, neighbor_cond], Q[row, neighbor_cond] = get_state_value(
                    V=V,
                    Q=Q,
                    P=P,
                    R=R,
                    row=row,
                    neighbor_cond=neighbor_cond,
                    gamma=gamma,
                    policy_improvement=False)

        delta = np.max(np.abs(V - oldV))
        iterCnt += 1
        #print(f'V{iterCnt}')
        #print(V)
    return V, Q, iterCnt


def policy_improvement(V, Q, P, R, gamma=1):
    """
    This function performs policy improvement using the action-value function for the optimal state-value function
    :param Q: Action-value function; Q contains the final updated values from value-iteration step.
    :return: 2D-matrix pi
    """
    # policy pi
    pi = np.zeros((Q.shape[0], Q.shape[1]))

    # Get optimal policy by being greedy in finding argmax a for state s
    for row in range(V.shape[0]):
        for neighbor_cond in range(V.shape[1]):
            # print("State")
            # print(f"row: {row+1} and neighbor: {neighbor_cond}")
            V[row, neighbor_cond], Q[row, neighbor_cond] = get_state_value(
                V=V,
                Q=Q,
                P=P,
                R=R,
                row=row,
                neighbor_cond=neighbor_cond,
                gamma=gamma,
                policy_improvement=True)

    # Loop through the matrix that contains the action-value function
    for j, row in enumerate(Q):
        for i, col in enumerate(row):
            best_action = np.argmax(col)
            pi[j, i] = best_action
    return pi


def policy_vs_gamma(N, M, A, P, R):
    # compare the optimal policy for different discount factors gamma
    gammas = np.linspace(0, 1, 101)  # generate gammas
    y = get_xlabels_barh(N=N, M=M)  # generate labels for the plot

    # compute the optimal state-value function and the optimal policy for different gammas
    for gamma in gammas:
        # Generate state space
        V_zeros = np.zeros((N, M))  # state_values
        Q_zeros = np.zeros((N, M, A))  # action_values

        # Do policy improvement through value iteration and policy improvement
        V, Q, iterCnt = value_iteration(V=V_zeros, Q=Q_zeros, P=P, R=R, gamma=gamma)
        pi = policy_improvement(V, Q, P, R, gamma=1)
        pi_flatten = pi.flatten()

        # plot the actions with different symbols
        for state_index, state_label in enumerate(y):
            if pi_flatten[state_index] == 0:
                plt.scatter(gamma, state_label, marker='o', c='k', s=6)
            elif pi_flatten[state_index] == 1:
                plt.scatter(gamma, state_label, marker='v', c='r', s=6)

    # Plot the optimal policies for different discount factors
    plt.figure(figsize=(16, 12))

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Action: stay', markerfacecolor='k'),
                       Line2D([0], [0], marker='v', color='w', label='Action: next', markerfacecolor='r')]

    plt.title('Optimal Policy for different Discount Factors Gamma', fontsize=20)
    plt.xlabel(f'Discount Factor $\gamma$', fontsize=16)
    plt.ylabel(f'State \n Current Row + Condition of Neighboring Seat', fontsize=16)
    plt.legend(handles=legend_elements, ncol=2)

    plt.show()
    plt.savefig('policy_vs_gamma.jpg')
    return

# Helper function to get labels for gamma vs optimal policy figure
def get_xlabels_barh(N, M):
    N_rows = N
    N_cols = M

    cond = ['Empty', 'Masked', 'Non-Masked']
    x = []
    for i in range(N_rows):
        for j in range(N_cols):
            curr_label = f'{N_rows - i} - {cond[j]}'
            x.append(curr_label)
    return x

# Function for analysis 2:
def runtime_vs_N(N_total, P, R):
    M = 3
    A = 2

    # Runtime is measured in iteration until convergence to the optimal state-value function / optimal policy
    runtime = []
    for N in range(1, N_total+1):
        print(f'Current Number of Rows: {N}')
        # Generate state space
        V = np.zeros((N, M))  # state_values
        Q = np.zeros((N, M, A))  # action_values
        V, Q, iterCnt = value_iteration(V=V, Q=Q, P=P, R=R, gamma=1)

        runtime.append(iterCnt)

    plt.figure(figsize=(12, 10))
    Ns = range(1, N_total+1)

    plt.plot(Ns, runtime)
    plt.title('Runtime in Epochs vs Number of Rows', fontsize=20)
    plt.xlabel('Number of Rows N', fontsize=16)
    plt.ylabel(f'Runtime in Epochs', fontsize=16)

    plt.savefig('runtime_vs_N.jpg')

    plt.show()
    return Ns, runtime


def main(argv):
    """
    Main function to run the program. Please use the following command in your command line to run the example setup:
    python myProgram.py N p_taken p_mask p_sick_empty p_sick_masked p_sick_nonmasked R_sick R_exit R_learn
    python ECE517_RL_MetznerChristoph_Project1.py 12 0.5 0.5 0.01 0.1 0.5 -100 -100 2
    or
    python3 ECE517_RL_MetznerChristoph_Project1.py 12 0.5 0.5 0.01 0.1 0.5 -100 -100 2

    :param argv: list of command line arguments
    :return: V - optimal state-value-function and pi - optimal policy give optimal state-value-function
    """

    N = int(argv[1])  # Number of rows specified by the user: 12
    M = 3  # Number of conditions of neighboring seat
    A = 2  # Two actions: stay and next

    # MDP Dynamics / Transition Probabilites:
    # Action: Take seat - 0
    p_sick_empty = float(argv[4])  # 0.01
    p_sick_masked = float(argv[5])  # 0.1
    p_sick_nonmasked = float(argv[6])  # 0.5

    # Action: Go to next row - 1
    p_taken = float(argv[2])  # 0.5
    p_mask = float(argv[3])  # 0.5

    # Probabilities specified by user
    P= np.array([p_taken, p_mask, p_sick_empty, p_sick_masked, p_sick_nonmasked])

    # Rewards
    # Rewards for Action: Take seat - 0
    R_sick = float(argv[7])  # -100
    R_learn = float(argv[9])  # 2

    # Rewards for Action: Got to next row - 1
    R_next = 0.0
    R_exit = float(argv[8])  # -100

    # Rewards given by the user: R
    R = [[R_sick, R_learn], [R_next, R_exit]]

    # Generate value-functions
    V = np.zeros((N, M))  # state_values
    Q = np.zeros((N, M, A))  # action_values

    # Call the function value_iteration to find the optimal state-values
    V, Q, iterCnt = value_iteration(V=V, Q=Q, P=P, R=R, gamma=1)

    # Call the function policy_improvement to find the optimal policy (optimal actions)
    pi = policy_improvement(V, Q, P, R, gamma=1)

    print('Optimal State-Value-Function V:')
    print(V)
    print('\nOptimal Policy - Pi:')
    print(pi)

    ##################################################################
    #### This part of the program is used for purpose of analysis ####
    ##################################################################

    # Analysis of algorithm
    # 1. Analysis - optimal policy vs different gammas for example setup
    #policy_vs_gamma(N, M, A, P, R)

    # 2. Analysis - run-time vs number of rows N
    # Ns, runtime = runtime_vs_N(N_total=100, P=P, R=R)

    # 3. Analysis - Effects of changes in dynamics of MDP on Policy
    # a) Case: Higher Probability to Get Sick - New Variant Emerged - Students are more likely to get sick in all states
    # python ECE517_RL_MetznerChristoph_Project1.py 12 0.5 0.5 0.1 0.3 0.7 -100 -100 2
    # b) Case: No Reward for leaving the class - The professor and class do not shame the exiting student :-)
    # python ECE517_RL_MetznerChristoph_Project1.py 12 0.5 0.5 0.01 0.1 0.5 -100 0 2
    # c) Case: Lower Negative Reward for getting sick
    # python ECE517_RL_MetznerChristoph_Project1.py 12 0.5 0.5 0.01 0.1 0.5 -30 -100 2

    return V, pi

if __name__ == '__main__':
    main(sys.argv)
