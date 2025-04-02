import numpy as np
import matplotlib.pyplot as plots
from itertools import chain

def flatten(l):
    return [item for sublist in l for item in sublist]

def value_iteration(transitions , reward , states, actions, g=0.9,eps=1e-3): 
    V1 = {x: 0 for x in range(states)}#initialize V
    while True:
        d = 0
        V = V1.copy()#update new V
        for s in range(states):
            clist = []
            for a in range(actions):
                b = enumerate(transitions[s, a, :])
                c = 0.0 
                for i in b:
                    c += i[1] * V[i[0]]
                clist.append(c)
            V1[s] = reward[s] + g * max(clist) #return maximum argument of [list of 4 actions]
            d = max(d, abs(V1[s] - V[s]))
        if d < eps * (1 - g) / g:
            return V

def utility(transitions,state,a, V):
    b = enumerate(transitions[state, a, :]) # [0. 0. 0. ... 0. 1. 0.] multiply 0*0.0 + .. + 15*1.0+16*0.0
    c = 0.0 
    for i in b:
        c += i[1] * V[i[0]] #probability * V(s') s':(previous state)
    return c

def optimal_policy(transitions, V):
    states, actions, _ = transitions.shape
    policy = {}  
    #for every state find optimal action
    for s in range(states):
        dlist = []
        policy[s] = max(range(actions), key=lambda a: utility(transitions,s,a, V))#compute utility for every action, return action with max value
    return policy

def plot_vit(V,gridworldvar):
    plot = np.zeros(gridworldvar.shape)
    for key, val in V.items():
        plot[key // gridworldvar.shape[1], key % gridworldvar.shape[1]] = val #plot heat map , increased value means brighter colour
    return plot

def plot_arrow(P, env , shapevar):#draw arrows based on best policy
        for key, val in P.items():
            if val == env.UP:
                plots.arrow(key // shapevar[1], key % shapevar[1], -0.30, 0, head_width=0.05)
            elif val == env.RIGHT:
                plots.arrow(key // shapevar[1], key % shapevar[1], 0, 0.30, head_width=0.05)
            elif val == env.DOWN:
                plots.arrow(key // shapevar[1], key % shapevar[1], 0.30, 0, head_width=0.05)
            elif val == env.LEFT:
                plots.arrow(key // shapevar[1], key % shapevar[1], 0, -0.30, head_width=0.05)

if __name__ == '__main__':
    from envs import gridworld
    gridworldvar = gridworld.GridworldEnv()
    
    alist = [gridworldvar.P[s] for s in range(gridworldvar.nS)]

    print(alist)

    #parse gym instance
    blist = []
    for x in alist:
        clist = []
        for y in x:
            z = flatten(x[y])
            clist.append(flatten(np.eye(1,gridworldvar.nS,z[1]))) # No. of rows , nS :number of states , index(second value: 1)
        blist.append(clist)

    transitions = np.array(blist)
    print(transitions)

    print('\n\n\n\n')

    print(np.transpose(transitions))

    #get reward from the dict
    alist = [gridworldvar.P[s][0][0][2] for s in range(gridworldvar.nS)]
    reward = np.array(alist) 
    print(reward)

    states,actions,key = transitions.shape
    # print(states)
    # print(transitions[0][0])
    # l1 = enumerate(transitions[0, 0, :])
    # for i in l1:
    #     print(i[1])

    V = value_iteration(transitions, reward,states,actions)
    op = optimal_policy(transitions, V)
   
    print(V)
    print(op)
    print("shape")
    print(transitions.shape)

    print(gridworldvar.shape)
    plot = plot_vit(V,gridworldvar)
    plots.matshow(plot)
    plots.savefig("value_iteration.png")

    plot_arrow(op,gridworld, gridworldvar.shape)
    plots.savefig("value_iteration_policy.png")
 
