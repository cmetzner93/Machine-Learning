"""
Title: Project #2 - Bomb away!
Author: Christoph Metzner
Date: 10-28-21
Course: ECE 517 - Reinforcement Learning
"""

# imported libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import List, Tuple
import numpy.typing as npt
import time


def init_starting_state(d: int) -> List[int]:
    """
    Function that initializes the starting state, i.e., starting position of robot and bomb. This function makes sure,
    that the bomb is not placed on the robot and the game ends prematurely.

    Parameters
    ----------
    d: int
        discrete number defining the dimension of the squared grid
    Returns
    -------
    starting_state: list
        List containing integer values for x/y positions of robot and bomb.
    """
    x_pos_robot = np.random.randint(low=0, high=d)
    y_pos_robot = np.random.randint(low=0, high=d)
    x_pos_bomb = np.random.randint(low=0, high=d)
    y_pos_bomb = np.random.randint(low=0, high=d)

    # Check if starting positions of robot and bomb are the same; if the same repeat init starting position of bomb
    while x_pos_robot == x_pos_bomb and y_pos_robot == y_pos_bomb:
        x_pos_bomb = np.random.randint(low=0, high=d)
        y_pos_bomb = np.random.randint(low=0, high=d)

    starting_state = [x_pos_robot, y_pos_robot, x_pos_bomb, y_pos_bomb]
    return starting_state


def get_action(pi: npt.ArrayLike, s: List[int]) -> str:
    """
    Function that selects action based on the policy for a given state.
    Parameters
    ----------
    pi: numpy array
        A stochastic policy pi containing probabilities for each possible action for a given state
    s: list of ints
        A list containing the state of the setup, i.e., positions of the robot and bomb

    Returns
    -------
    str
        selected action for the given state

    """
    x_pos_robot = s[0]
    y_pos_robot = s[1]
    x_pos_bomb = s[2]
    y_pos_bomb = s[3]

    # This gives a list with the respective probabilities per action for current state
    action_probabilities_state = pi[x_pos_robot, y_pos_robot, x_pos_bomb, y_pos_bomb]

    # select an action based on the probability
    action = np.random.choice(['north', 'south', 'east', 'west'],
                              p=action_probabilities_state)

    return action


def get_next_state(s: List[int], action: str, d: int) -> Tuple[List[int], bool]:
    """
     Function that computes the next state. The order of computation is important.
    1) Robot Moves, 2) Check if pos_robot equals pos_bomb, if true then 3) Bomb moves
    Parameters
    ----------
    s: List[int]
        A list containing the state of the setup, i.e., positions of the robot and bomb
    action: str
        Selected action of the robot
    d: int
        dimension of the squared grid

    Returns
    -------
    s_prime: List[int]
    """

    # Clarifying idx's
    # values of current state/pos of robot
    # s[0] = y_pos_robot
    # s[1] = x_pos_robot
    # s[2] = y_pos_bomb
    # s[3] = x_pos_bomb

    # Set current state s equal to successor state s' to update state
    s_prime = s.copy()
    # boolean values indicate whether the robot is at a terminal state or not
    if action == 'north':
        # Check if move is out of bounds: If true then return previous state, else move one step
        # If out of bounds, the values of the old state s are returned as the successor state s'
        if s[0] - 1 < 0:  # y-position of robot
            return s, False
        else:
            s_prime[0] = s[0] - 1
    # Check if move is out of bounds: If true then return previous state, else move one step
    # If out of bounds, the values of the old state s are returned as the successor state s'
    elif action == 'south':
        if s[0] + 1 == d:
            return s, False
        else:
            s_prime[0] = s[0] + 1
    # Check if move is out of bounds: If true then return previous state, else move one step
    # If out of bounds, the values of the old state s are returned as the successor state s'
    elif action == 'east':
        if s[1] + 1 == d:
            return s, False
        else:
            s_prime[1] = s[1] + 1
    # Check if move is out of bounds: If true then return previous state, else move one step
    # If out of bounds, the values of the old state s are returned as the successor state s'
    elif action == 'west':
        if s[1] - 1 < 0:
            return s, False
        else:
            s_prime[1] = s[1] - 1

    # Check if new position of robot (s') is equal to position of bomb from current state (s)
    if s_prime[0] == s[2] and s_prime[1] == s[3]:  # if y_pos_robot == y_pos_bomb and x_pos_robot == x_pos_bomb
        if action == 'north':
            if s[2] - 1 < 0:  # check if bomb is pushed outside for grid --> terminate episode
                return s, True
            else:  # else move bomb in y direction
                s_prime[2] = s[2] - 1
                return s_prime, False
        elif action == 'south':  # check if bomb is pushed outside for grid --> terminate episode
            if s[2] + 1 == d:
                return s, True
            else:  # else move bomb in y direction
                s_prime[2] = s[2] + 1
                return s_prime, False
        elif action == 'east':  # check if bomb is pushed outside for grid --> terminate episode
            if s[3] + 1 == d:
                return s, True
            else:  # else move bomb in x direction
                s_prime[3] = s[3] + 1
                return s_prime, False
        elif action == 'west':  # check if bomb is pushed outside for grid --> terminate episode
            if s[3] - 1 < 0:
                return s, True
            else:  # else move bomb in x direction
                s_prime[3] = s[3] - 1
                return s_prime, False
    else:  # if position of robot and bomb not equal than return s_prime and False for terminal state
        return s_prime, False


def get_reward(s: List[int], action: str, s_prime: List[int], d: int, terminal: bool, r_struct: int) -> int:
    """
    Function that calculates the reward for a given move and state and selected reward structure.
    Parameters
    ----------
    s: List[int]
        List containing current state information
    action: str
        selected action
    s_prime: List[int]
        List containing successor state information
    d: int
        Dimension of the squared grid
    terminal: bool
        variable holding information about the condition of the simulation
    r_struct: int
        integer indicating currently selected reward structure for learning

    Returns
    -------
    int
        reward for that action
    """
    if r_struct == 0:  # -1 when robot moves
        return -1
    elif r_struct == 1:  # -1 when robot moves, +1 if bomb is moved away from center, +10 if bomb is pushed into river
        # Here we have to check for three cases
        # Reward for moving the robot
        r_move_robot = -1
        # Reward for moving the bomb or pushing the bomb into the river
        if terminal is True:  # when terminal is true than robot has pushed the bomb into the river
            r_bomb = 10
        else:
            r_bomb = 0
            # check if bomb has been moved
            if s[2] != s_prime[2] and s[3] != s_prime[3]:
                # Get centerline to indicate whether the bomb was moved to or away from it
                if d % 2 > 0:  # uneven
                    # the next line gives us the center row/column index to determine if bomb is moved to or away
                    center_line = int(np.floor(d/2))  # Example: d=7, matrix idx: 0,1,2,  3  ,4,5,6
                else:
                    center_line = d/2  # Example: d=8, matrix idx: 0,1,2,3,     4,5,6,7 --> 3.5, so if s' further
                    # away than s to 3.5 --> R=+1
                if action == 'north' or action == 'south':
                    d_s = np.abs(center_line - s[2])  # compute distance of y_pos_bomb to centerline for s and s'
                    d_s_prime = np.abs(center_line - s_prime[2])
                    if d_s_prime > d_s:
                        r_bomb = 1
                elif action == 'east' or action == 'west':
                    d_s = np.abs(center_line - s[3])  # compute distance of x_pos_bomb to centerline for s and s'
                    d_s_prime = np.abs(center_line - s_prime[3])
                    if d_s_prime > d_s:
                        r_bomb = 1
        r = r_move_robot + r_bomb  # compute total reward for the move
        return r


def get_pi(Q_s: List[float], e: float) -> List[float]:
    """
    Function that updates the stochastic epsilon-greedy policy
    Parameters
    ----------
    Q_s: List[float]
        List containing all the action-values for the current state-action-pair
    e: float
        value for epsilon greedy, i.e., determines degree of exploration

    Returns
    -------
    numpy array
        Updated policy for that state

    """
    # lets account for multiple maximum values (especially when a state hasnt been visited)
    pi_s = [0, 0, 0, 0]  # initialize array holding probabilities
    max_Qs = np.max(Q_s)  # get maximum Q-value for update
    count_max = Q_s.tolist().count(max_Qs)  # determine number of maximum q_values
    if count_max == 4:
        return [0.25, 0.25, 0.25, 0.25]
    else:
        for idx, Q_value in enumerate(Q_s):  # loop through q-values for updating probabilities
            if Q_value == max_Qs:
                pi_s[idx] = (1-e)/count_max
            else:
                pi_s[idx] = e/(4-count_max)  # 4 represents the number of possible actions
    return pi_s


def do_MC_learning(
        Q: npt.ArrayLike,
        pi: npt.ArrayLike,
        s: List[int],
        a: float,
        d: int,
        e: float,
        r_struct: int) -> Tuple[int, npt.ArrayLike, npt.ArrayLike]:
    """
    Function that performs one episode of epsilon-greedy on-line monte carlo learning.

    Parameters
    ----------
    Q: Arraylike
        Matrix containing all action-values for all states
    pi: Array like
        Matrix containing all probabilities for the actions following the stochastic epsilon-greedy policy
    s: List[int]
        List containing information about the current state
    a: float
        value for learning rate / step-size
    d: int
        dimension of the squared grid
    e: float
        value for epsilon
    r_struct: int
        determines reward structure used for learning

    Returns
    -------
    return_episode: int
        total return for episode
    pi: array like
        Updated matrix containing all probabilities for the actions following the stochastic epsilon-greedy policy
    """
    MaxIterCnt = 1000  # limit number of moves
    IterCnt = 0  # count number of moves
    terminal = False  # set terminal state of game to False
    action_idx = {'north': 0, 'south': 1, 'east': 2, 'west': 3}

    episode = []
    # Generate trajectory for episode
    while terminal is False and IterCnt < MaxIterCnt:
        # call function to get new action
        action = get_action(pi=pi, s=s)
        # call function to get new state or reached terminal state
        s_prime, terminal = get_next_state(s=s, action=action, d=d)
        # call function to get reward
        r = get_reward(s=s, action=action, s_prime=s_prime, d=d, terminal=terminal, r_struct=r_struct)
        episode.append((s, action, r))
        s = s_prime
        IterCnt += 1  # +1 move

    # Need to reverse list since we start updating from terminal state
    episode.reverse()
    return_episode = 0  # expected return
    for t, state_action in enumerate(episode):
        s = state_action[0]  # state
        action = state_action[1]  # action
        r = state_action[2]  # reward
        return_episode += r  # add reward to current return
        Q_sa = Q[s[0], s[1], s[2], s[3], action_idx[action]]  # Get action-value of current state-action pair
        Q[s[0], s[1], s[2], s[3], action_idx[action]] = Q_sa + a * (return_episode - Q_sa)  # update Q
        pi[s[0], s[1], s[2], s[3]] = get_pi(Q_s=Q[s[0], s[1], s[2], s[3]], e=e)  # update policy
    return return_episode, pi


def do_Q_learning(
        Q: npt.ArrayLike,
        pi: npt.ArrayLike,
        s: List[int],
        a: float,
        d: int,
        e: float,
        r_struct: int) -> Tuple[int, npt.ArrayLike, npt.ArrayLike]:
    """
    Function that performs one episode of epsilon-greedy Q-learning.

    Parameters
    ----------
    Q: Arraylike
        Matrix containing all action-values for all states
    pi: Array like
        Matrix containing all probabilities for the actions following the stochastic epsilon-greedy policy
    s: List[int]
        List containing information about the current state
    a: float
        value for learning rate / step size
    d: int
        dimension of the squared grid
    e: float
        value for epsilon
    r_struct: int
        determines reward structure used for learning

    Returns
    -------
    return_episode: int
        total return for episode
    pi: array like
        Updated matrix containing all probabilities for the actions following the stochastic epsilon-greedy policy
    Q: array like
        updated matrix containing all action-values for all states
    """
    MaxIterCnt = 1000  # limit number of moves
    IterCnt = 0  # count number of moves
    terminal = False  # set terminal state of game to False
    return_episode = 0
    action_idx = {'north': 0, 'south': 1, 'east': 2, 'west': 3}

    # move through one episode
    while terminal is False and IterCnt < MaxIterCnt:
        # call function to get new action
        action = get_action(pi=pi, s=s)
        # call function to get new state or reached terminal state
        s_prime, terminal = get_next_state(s=s, action=action, d=d)
        # call function to get reward
        r = get_reward(s=s, action=action, s_prime=s_prime, d=d, terminal=terminal, r_struct=r_struct)

        # Update Q-Value
        Q_sa = Q[s[0], s[1], s[2], s[3], action_idx[action]]  # Get action-value of current state-action pair
        max_Q_sprime_a = np.max(Q[s_prime[0], s_prime[1], s_prime[2], s_prime[3]])  # Get max Q-value for s'-a pair
        Q[s[0], s[1], s[2], s[3], action_idx[action]] = Q_sa + a * (r + max_Q_sprime_a - Q_sa)
        # Update probabilities in policy using epsilon greedy approach
        pi[s[0], s[1], s[2], s[3]] = get_pi(Q_s=Q[s[0], s[1], s[2], s[3]], e=e)
        return_episode += r
        s = s_prime
        IterCnt += 1
    return return_episode, pi, Q


def execute_learning(
        m: int,
        d: int,
        n: int,
        a: float,
        e: float,
        r_struct: int) -> Tuple[List[int], npt.ArrayLike]:
    """
    Function that controls and performs learning by calling either function for MC or Q-learning over n episodes.

    Parameters
    ----------
    m: int
        Variable indicating selected learning method: 1 - MC learning and 2 - Q-learning
    d: int
        dimension of squared grid
    n: int
        number of epochs
    a: float
        value for learning rate / step size
    e: float
        value for epsilon
    r_struct: int
        variable indicating reward structure for learning

    Returns
    -------
    total_returns: List[int]
        total returns for all episodes
    pi: array like
        Updated matrix containing all probabilities for the actions following the stochastic epsilon-greedy policy

    """
    Q = np.zeros((d, d, d, d, 4))  # Initialize action-values Q
    pi = np.full((d, d, d, d, 4), 0.25)  # Initialize starting pi with uniform distribution
    # check for learning method
    total_returns = []
    if m == 1:  # do epsilon-greedy online Monte Carlo with incremental updates
        # Loop for each episode of training
        for episode in range(n):
            #print(f'Epoch: {episode+1}')
            starting_state = init_starting_state(d=d)
            total_return, pi = do_MC_learning(Q=Q, pi=pi, s=starting_state, a=a, d=d, e=e, r_struct=r_struct)
            total_returns.append(total_return)
    elif m == 2:  # do Q-Learning
        # Loop for each episode of training
        for episode in range(n):
            #print(f'Epoch: {episode+1}')
            # index of starting state / state: [x_pos_robot, y_pos_robot, x_pos_bomb, y_pos_bomb]
            starting_state = init_starting_state(d=d)
            total_return, pi, Q = do_Q_learning(Q=Q, pi=pi, s=starting_state, a=a, d=d, e=e, r_struct=r_struct)
            total_returns.append(total_return)
            #print(f'Probabilities of action-values for initial setup:\n{pi[5, 0, 4, 1]}')
            if (episode+1) % 2000 == 0:
                print(f'Epoch: {episode+1}')
                print(f'Action-Values for initial Setup:\n{Q[5, 0, 4, 1]}')
    return total_returns, pi


def plot_episode(d, s, pi):
    # A function which plots an episode given a starting state. This will allow you to view episodes
    # after training.
    """
    A function that plots an episode given a starting date and a trained policy.
    Parameters
    ----------
    d: int
        discrete number indicating dimension of the squared grid
    s: list
        list containing integers representing x- and y-positions of robot and bomb, i.e., the starting state
    pi: numpy array
        matrix containing the policy for the robot
    """
    # Generate trajectory given starting state s using learned stochastic policy pi
    MaxIterCnt = 1000  # limit number of moves
    IterCnt = 0  # count number of moves
    terminal = False  # set terminal state of game to False
    trajectory = []
    while terminal is False and IterCnt < MaxIterCnt:
        action = get_action(pi=pi, s=s)  # call function to get new action
        # call function to get new state or reached terminal state
        s_prime, terminal = get_next_state(s=s, action=action, d=d)
        trajectory.append((s, action))
        s = s_prime
        IterCnt += 1

    print(trajectory)

    # Plot the trajectory
    # Initialize grid and positions of the robot and bomb
    for t, state_action in enumerate(trajectory):
        grid = np.zeros((d, d))
        action = state_action[1]
        grid[state_action[0][0]][state_action[0][1]] = 1  # position of robot
        grid[state_action[0][2]][state_action[0][3]] = 9  # position of bomb
        print(f's{t}:\n{grid}')
        print(f'a{t}: {action}')

        # show terminal state of the simulation
        if t+1 == len(trajectory):
            grid = np.zeros((d, d))
            last_state = trajectory[-1][0]
            grid[last_state[2]][last_state[3]] = 1  # position of robot
            print(f's{t + 1}:\n{grid}')


def plot_total_returns(total_returns: List[int], label: int, color: str, reward_struct: int):
    x = np.linspace(1, len(total_returns), len(total_returns))
    plt.figure(figsize=(10, 8))

    if label == 1:
        label = 'Epsilon-greedy on-line Monte Carlo'
    else:
        label = 'Q-Learning'

    if reward_struct == 0:
        reward_struct = '1'
    else:
        reward_struct = '2'

    plt.plot(x, total_returns, label=label, color=color)
    plt.title(f'Total Return vs Epoch\n {label} with reward structure {reward_struct}')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Total Return per Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig(f'Total Return vs Epoch {label} and {reward_struct}.png')



def main(argv):
    """
    Run the example setup like
    python file.py d a e n m
    python ece517_RL_project2.py 8 0.1 0.1 20000 2
    """
    d = int(argv[1])  # dimension of the grid
    a = float(argv[2])  # value for learning rate alpha
    e = float(argv[3])  # value for epsilon
    n = int(argv[4])  # total number of episodes to train on
    m = int(argv[5])  # learning method: MC (1) and Q-Learning (2)
    r1 = 0  # only -1 when robot moves
    r2 = 1  # -1 when robot moves, +1 if bomb is moved away from center, +10 if bomb is pushed into river

    # Plot moves of robot for example setup for different policies:
    starting_state = [5, 0, 4, 1]  # [y_pos_robot, x_pos_robot, y_pos_bomb, x_pos_bomb]

    # Reward Structure: 1
    start_time = time.time()
    total_returns_r1, pi_r1 = execute_learning(m=m, d=d, n=n, a=a, e=e, r_struct=r1)
    end_time = time.time()
    print(f'Total elapsed time for training in seconds: {end_time - start_time}')
    plot_episode(d=d, s=starting_state, pi=pi_r1)
    plot_total_returns(total_returns=total_returns_r1, label=m, color='b', reward_struct=r1)


    # Reward Structure: 2
    start_time = time.time()
    total_returns_r2, pi_r2 = execute_learning(m=m, d=d, n=n, a=a, e=e, r_struct=r2)
    end_time = time.time()
    print(f'Total elapsed time for training in seconds: {end_time - start_time}')
    plot_episode(d=d, s=starting_state, pi=pi_r2)
    plot_total_returns(total_returns=total_returns_r2, label=m, color='r', reward_struct=r2)




if __name__ == "__main__":
    main(sys.argv)
