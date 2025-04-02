from envs import gridworld
import numpy as np
from value_iteration import *

def plot_res(V,gridworldvar):
    plot = np.zeros(gridworldvar.shape)
    for key, val in enumerate(V):
        plot[key // gridworldvar.shape[1], key % gridworldvar.shape[1]] = val #plot heat map , increased value means brighter colour
    return plot

def generate_trajectories(policies, envr, trajectories_len=5, trajectories_num=250):
    trajectories = []
    for _ in range(trajectories_num):
        step = []
        for j in range(trajectories_len):
            current_state = envr.s
            # # Take the action and get the new observation space
            statev, rewardv, donev, info = envr.step(policies[current_state])
            vart = (current_state, policies[current_state], statev)
            step.append(vart)
            #if we reach goal state fill the remaining steps of the trajectory with that state.
            if donev:
                for _ in range(j + 1, trajectories_len):
                    var = (statev, 0, statev)
                    step.append(var)
                break
        envr.reset()
        trajectories.append(step)
    return trajectories

"""
transitions: A 3d array mapping (state_a, action, state_b) to
        the probability of transitioning from state_a to state_b with action.
        Shape (states, actions, states).
"""
def state_visitation_frequency(transitions, policies ,trajectories,states_num):
    trajectories_num = len(trajectories[0])
    svf = np.zeros((states_num, trajectories_num))

    #get start state for first step from every trajectory
    for trjs in trajectories:
        svf[trjs[0][0], 0] = svf[trjs[0][0], 0] + 1
    
    #loop for every trajectory updating the value for every state
    for trj in range(1, trajectories_num):
        for st in range(states_num):
            svf_var = []
            for prev_st in range(states_num):
                svf_var.append( transitions[prev_st, policies[prev_st], st] * svf[prev_st, trj - 1])
            svf[st, trj] = sum(svf_var)
    #state visitation frequency
    #the (discounted) sum of probabilities of visiting a given state.
    #get the mean svf 
    return np.sum(svf,axis=1)/ len(trajectories)
            
def maximumentropy_irl(features_array, trajectories, transitions ,states,actions,
                learning_rate=0.5, g=0.9, epochs=15):
    """
    The nth row of the matrix represents the nth state. 
    Feature matrix is an identity matrix in this case(the place of 1 [0-15] means the No.state) , but I treat it
    as a shape (N, D) where N is the number of states and D is the
    dimensionality of that specific state. 
    """
    states_num, states_dim = features_array.shape
    thita = np.random.uniform(size=(states_dim))

    feature_expectation = np.zeros(states_dim)#(1,states) size
    for trajectory in trajectories:
        for step in trajectory:#for every state,action pair (ζ)
            #choose the step[0] row of the I(identity) matrix (basically count the frequency of all current states)
            feature_expectation = feature_expectation + features_array[step[0], :]#φ(ζ) : sum of state features along the path
            #increase the specific state value

    #average path
    feature_expectation /= len(trajectories)# Φ

    for _ in range(epochs):
        reward = features_array.dot(thita)
        V = value_iteration(transitions, reward,states,actions, g)
        op = optimal_policy(transitions, V)
        svisit_fq = state_visitation_frequency(transitions, op , trajectories,states_num)
        gradient = feature_expectation - features_array.dot(svisit_fq)
        thita = thita + learning_rate * gradient 

    return features_array.dot(thita)#return reward

if __name__ == '__main__':
 
    gridworldvar = gridworld.GridworldEnv()
    
    alist = [gridworldvar.P[s] for s in range(gridworldvar.nS)]

    #parse gym instance
    blist = []
    for x in alist:
        clist = []
        for y in x:
            z = flatten(x[y])
            clist.append(flatten(np.eye(1,gridworldvar.nS,z[1]))) # No. of rows , nS :number of states , index(second value: 1)
        blist.append(clist)

    transitions = np.array(blist)

    print('\n\n\n\n')

    #get reward from the dict
    alist = [gridworldvar.P[s][0][0][2] for s in range(gridworldvar.nS)]
    reward = np.array(alist) 
    #print(reward)

    states,actions,key = transitions.shape

    V = value_iteration(transitions, reward, states, actions)
    op = optimal_policy(transitions, V)

    #print(op)
    #print(transitions)
    #print(trajectories)
    
    trajectories = generate_trajectories(op, gridworldvar)

    result = maximumentropy_irl(np.eye(gridworldvar.nS), trajectories, transitions ,states,actions)
    print(result)

    plot = plot_res(result,gridworldvar)
    plots.matshow(plot)
    plots.savefig("maximumentropy_irl.png")

