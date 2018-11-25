"""
Solving FrozenLake8x8 environment using Value-Itertion.


Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
import random as rand
import time
from qlearner import QLearner


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.

    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.

    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma = 1.0):
    """ Extract the policy given a value-function """
    number_of_states = env.unwrapped.nS

    policy = np.zeros(number_of_states)
    for s in range(number_of_states):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.unwrapped.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """

    number_of_states = env.unwrapped.nS
    number_of_actions = env.unwrapped.nA
    policy = env.unwrapped.P

    v = np.zeros(number_of_states)  # initialize value-function
    max_iterations = 100000
    converged_at = 0
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(number_of_states):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in policy[s][a]]) for a in range(number_of_actions)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            converged_at = i + 1
            #print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v, converged_at


def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    number_of_states = env.unwrapped.nS
    v = np.zeros(number_of_states)
    env_policy = env.unwrapped.P

    eps = 1e-10
    i = 0
    while True:

        prev_v = np.copy(v)
        for s in range(number_of_states):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env_policy[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
        else:
            i += 1

    return v, i


def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    number_of_states = env.unwrapped.nS
    number_of_actions = env.unwrapped.nA

    policy = np.random.choice(number_of_actions, size=(number_of_states))  # initialize a random policy
    max_iterations = 200000

    policy_convergence = []

    converged_at = 0
    for i in range(max_iterations):
        old_policy_v, policy_convergence_v = compute_policy_v(env, policy, gamma)
        policy_convergence.append(policy_convergence_v)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            converged_at = i +1
            #print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy, converged_at, np.mean(policy_convergence)


def get_agent_pos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 'S':
                C = col
                R = row
    if (R+C)<0:
        print "warning: start location not defined"
    return R, C


def get_goal_pos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 'G':
                C = col
                R = row
    if (R+C) < 0:
        print "warning: goal location not defined"
    return R, C


# convert the location to a single integer
def discretize(pos):
    return pos[0]*10 + pos[1]


# move the robot and report reward
def move_bot(data,oldpos,a):
    testr, testc = oldpos

    randomrate = 0.20  # how often do we move randomly
    quicksandreward = -100  # penalty for stepping on quicksand

    # decide if we're going to ignore the action and
    # choose a random one instead
    if rand.uniform(0.0, 1.0) <= randomrate:  # going rogue
        a = rand.randint(0, 3)  # choose the random direction

    # update the test location
    if a == 0: #north
        testr = testr - 1
    elif a == 1: #east
        testc = testc + 1
    elif a == 2: #south
        testr = testr + 1
    elif a == 3: #west
        testc = testc - 1

    reward = -1 # default reward is negative one
    # see if it is legal. if not, revert
    if testr < 0: # off the map
        testr, testc = oldpos
    elif testr >= data.shape[0]: # off the map
        testr, testc = oldpos
    elif testc < 0: # off the map
        testr, testc = oldpos
    elif testc >= data.shape[1]:  # off the map
        testr, testc = oldpos

    elif data[testr, testc] == 'H':  # it is quicksand
        reward = quicksandreward
        data[testr, testc] = '6'  # mark the event

    elif data[testr, testc] == '6':  # it is quicksand
        reward = quicksandreward
        data[testr, testc] = '6'  # mark the event

    elif data[testr, testc] == 'G':  # it is the goal
        reward = 1 # for reaching the goal

    return (testr, testc), reward  # return the new, legal location


# print out the map
def print_map(data):
    print "--------------------"
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):

            if (row == 0 and col == 0) or data[row, col] == 'S':
                print "S",
            else :
                if data[row, col] == 'F':  # Frozen
                    print "F",
                if data[row, col] == 'H':  # Hole
                    print "H",
                if data[row, col] == 'G':  # Goal
                    print "G",

                if data[row,col] == '2': # El roboto
                    print "*",

                if data[row,col] == '4': # Trail
                    print ".",
                if data[row,col] == '5': # Quick sand
                    print "~",
                if data[row,col] == '6': # Stepped in quicksand
                    print "X",
        print
    print "--------------------"


def run_qlearner_test(map, epochs, learner, verbose):
    # each epoch involves one trip to the goal

    start_pos = get_agent_pos(map)  # find where the robot starts
    goal_pos = get_goal_pos(map)  # find where the goal is
    test_scores = np.zeros((epochs, 1))
    test_data = []
    test_time = []
    test_steps = []

    for epoch in range(1,epochs+1):

        start_time = time.time()
        total_reward = 0
        data = map.copy()
        robo_pos = start_pos
        state = discretize(robo_pos)  # convert the location to a state
        action = learner.querysetstate(state)  # set the state and get first action
        count = 0
        while (robo_pos != goal_pos) & (count < 10000):
            # move to new location according to action and then get a new action
            new_pos, step_reward = move_bot(data, robo_pos, action)
            if new_pos == goal_pos:
                r = 1  # reward for reaching the goal
            else:
                r = step_reward  # negative reward for not being at the goal
            state = discretize(new_pos)
            action = learner.query(state, r)

            if data[robo_pos] != '6':
                data[robo_pos] = '4'  # mark where we've been for map printing
            if data[new_pos] != '6':
                data[new_pos] = '2'  # move to new location
            robo_pos = new_pos  # update the location

            # if verbose: time.sleep(1)
            total_reward += step_reward
            count = count + 1

        end_time = time.time()

        if count == 100000:
            print "timeout"

        if verbose: print_map(data)
        if verbose: print epoch, total_reward

        test_scores[epoch-1, 0] = total_reward
        test_data.append(data)
        test_time.append(end_time - start_time)
        test_steps.append(count)

    best_epoch = np.argmax(test_scores)
    worst_epoch = np.argmin(test_scores)

    return_payload = np.median(test_scores), \
                    test_scores[best_epoch], test_data[best_epoch], test_time[best_epoch], test_steps[best_epoch], \
                    test_scores[worst_epoch], test_data[worst_epoch], test_time[worst_epoch], test_steps[worst_epoch],

    return return_payload

def test_qlearner(env, epochs, gamma):
    learner = QLearner(num_states=100,
                      num_actions=env.unwrapped.nA,
                      alpha=0.2,
                      gamma=gamma,
                      rar=0.5,
                      radr=0.999,
                      dyna=0,
                      verbose=False)

    return run_qlearner_test(env.unwrapped.desc, epochs, learner, False)



def run_experiment(env_name, gamma):
    test_env = gym.make(env_name)


    value_start = time.time()
    optimal_v, value_converged_at = value_iteration(test_env, gamma);
    value_end = time.time()
    policy = extract_policy(test_env, optimal_v, gamma)

    policy_start = time.time()
    optimal_policy, policy_converged_at, mean_value_iterations = policy_iteration(test_env, gamma=gamma)
    policy_end = time.time()

    epochs = 500

    print ""
    print '{}'.format(env_name)
    print '--------------------'
    print 'Value iteration converged at {} in {}'.format(value_converged_at, value_end - value_start)
    print 'Policy iteration converged at {} in {}; average policy computation iterations {}'.format(policy_converged_at, policy_end - policy_start, mean_value_iterations)
    print 'Value Iteration Policy  {}'.format(policy)
    print 'Policy Iteration Policy {}'.format(optimal_policy)
    print 'Same Policies? {}'.format(policy == optimal_policy)

    print 'Value Iteration average score = {}'.format(evaluate_policy(test_env, policy, gamma, n=epochs))
    print 'Policy Iteration average score = {}'.format(evaluate_policy(test_env, optimal_policy, gamma, n=epochs))


    median_reward, best_score, best_path, best_time, best_steps, worst_score, worst_path, worst_time, worst_steps = \
        test_qlearner(test_env, epochs, gamma)

    print "-------"
    print epochs, "median total_reward", median_reward
    print ""
    print '{} map'.format(env_name)
    print print_map(test_env.unwrapped.desc)
    print ""
    print 'best score: {} took {} steps in {}'.format(int(best_score), best_steps, best_time)
    print print_map(best_path)
    print ""
    print 'worst score: {} took {} steps in {}'.format(int(worst_score), worst_steps, worst_time)
    print print_map(worst_path)

if __name__ == '__main__':

    gamma = 0.9
    run_experiment('FrozenLake-v0', gamma)
    run_experiment('FrozenLake8x8-v0', gamma)

