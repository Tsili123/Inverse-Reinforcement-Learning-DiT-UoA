def plot_res(V,gridworldvar):
    plot = np.zeros(gridworldvar.shape)
    for key, val in enumerate(V):
        plot[key // gridworldvar.shape[1], key % gridworldvar.shape[1]] = val #plot heat map , increased value means brighter colour
    return plot

#T function return array 1xstates size
def T(tp, policy,eyes_2s,g ,a, s):
        return np.dot(tp[policy[s], s] - tp[a, s], np.linalg.inv(eyes_2s - g * tp[policy[s],:,:])) #because we have identity matrix , we need tp[policy[s]]

def lp_irl(trans_probs, policy, g=0.3, l1=1.4,Rmax=4.8):
    """
    Inverse Reinforcement learning with linear programming
    """
    states, actions, _ = trans_probs.shape
    action_set = set(range(actions))
    zero_1s = np.zeros(states)
    zero_2s = np.zeros((states, states))
    one_1s = np.transpose(np.ones(states))
    eyes_2s = np.eye(states)
    tp = np.transpose(trans_probs, (1, 0, 2))

    alist = []
    for s in range(states) :
        for a in action_set - {policy[s]}:
            alist.append(np.eye(1,states,s))
    I_stackmat = np.vstack(alist)

    blist = []
    for s in range(states) :
        for a in action_set - {policy[s]}:
            blist.append(-T(tp, policy,eyes_2s,g, a, s))
    T_stackmat = np.vstack(blist) #(n_states*n-1_actions)*nstates

    zero_stackmat = np.zeros((states * (actions - 1), states))

    A_ub = np.bmat([[T_stackmat, zero_stackmat, zero_stackmat], # -TR <= 0
                    [T_stackmat, I_stackmat, zero_stackmat],  # TR>=t => TR-t>=0 => -TR +t<=0
                    [-eyes_2s, zero_2s, -eyes_2s],   # -R <= u
                    [eyes_2s, zero_2s, -eyes_2s],  # R <= u
                    [eyes_2s, zero_2s, zero_2s], # R <= Rmax
                    [-eyes_2s, zero_2s, zero_2s],  # -R <= Rmax
                    ])  
    b_ub = np.vstack([np.zeros((  2 * states + states * (actions-1) * 2, 1)),
                   Rmax * np.ones((2 * states, 1))])

    c = -np.r_[zero_1s, one_1s, -l1 * one_1s]

    results = linprog(c, A_ub, b_ub)

    return results["x"][:states]

if __name__ == '__main__':
    from envs import gridworld
    from value_iteration import *
    from scipy.optimize import linprog
    import numpy as np


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
    #print(transitions)

    print('\n')

    #print(np.transpose(transitions))

    #get reward from the dict
    alist = [gridworldvar.P[s][0][0][2] for s in range(gridworldvar.nS)]
    reward = np.array(alist) 
    #print(reward)

    states,actions,key = transitions.shape

    V = value_iteration(transitions, reward,states,actions)
    op = optimal_policy(transitions, V)

    #print(op)
    #print(transitions)

    result = lp_irl(transitions, op,states , actions)
    print("Estimating the reward with linear programming:")
    print(result)

    plot = plot_res(result,gridworldvar)
    plots.matshow(plot)
    plots.savefig("linear_irl.png")

