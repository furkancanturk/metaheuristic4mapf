import sys
import argparse
import json
import yaml
import random
from math import fabs, exp
import pandas as pd
import numpy as np

from time import time 
from itertools import count, combinations, permutations
from functools import reduce
from copy import deepcopy, copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from pandas import ExcelWriter
from a_star import AStar
from path_objects import *

sys.path.insert(0, '../')
random.seed(1)
np.random.seed(1)
paths_dict = None

def path_dissimilarity(p1, p2, gamma):
    len_p1 = len(p1)
    len_p2 = len(p2)
    
    if len_p1 == len_p2:
        diff = [p1[t].location - p2[t].location for t in range(len_p1)]
        
    elif len_p1 > len_p2:
        extended_p2 = p2 + [p2[-1].move(t) for t in range(1, len_p1-len_p2+1)] 
        diff = [p1[t].location - extended_p2[t].location for t in range(len_p1)]
    else:
        extended_p1 = p1 + [p1[-1].move(t) for t in range(1, len_p2-len_p1+1)] 
        diff = [extended_p1[t].location - p2[t].location for t in range(len_p2)]
    
    distances = sorted([(_loc.x**2 + _loc.y**2)**0.5 for _loc in diff])[:gamma]
    
    return np.mean(distances)
     
def generate_plan(solution): # converts the solution form to write it on a file
    plan = {}
    for agent, path in solution.items():
        path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path]
        plan[agent] = path_dict_list
    return plan

def print_obj_lst(lst):
    print([str(obj) for obj in lst])


class Environment(object):
    def __init__(self, dimension, agents, obstacles=[], input_type = 0):
        global paths_dict
        self.input_type = input_type
        self.dimension = dimension
        self.obstacles = list(map(tuple, obstacles)) if obstacles else []
        
        self.n_free_vertices = dimension[0] * dimension[1] - len(obstacles) 

        self.agent_dict = self.make_agent_dict(agents) ## keeps start-goal states of each agent
        self.agent_names = list(self.agent_dict.keys())
        self.n_agents = len(self.agent_names)
        self.agent_codes = {name: name.replace("agent", "a") for name in self.agent_names} # to be used to generate solution code
        self.infeasible_opt_paths = {}

        self.current_solution_df = pd.DataFrame(columns=self.agent_names) #paths until t_max
        self.current_constraints = [] 
        self.current_reached_goals = [] # used for constructive feasible solution BUT ITS USAGE WILL BE UPDATED 
        self.current_tabu = []
        
        self.actions = [Location(0,0), Location(0,1), Location(0,-1), Location(-1,0), Location(1,0)]
        paths_dict = PathsDict(self.agent_names)
        
        density = round(len(self.obstacles)/self.dimension[0]/self.dimension[1]*100, 2)
        
        print(f"Map size: {dimension[0]}x{dimension[1]} | #Free Cells: {self.n_free_vertices} | Obstacle density%: {density} | #agents: {self.n_agents}" )
      
    def make_agent_dict(self, agents):
        agent_dict = {}
        for agent in agents:
            name = agent['name']
            start = agent['start']
            goal = agent['goal']
            if tuple(start) not in self.obstacles and tuple(goal) not in self.obstacles: 
                start_state = State(0, Location(start[0], start[1]))
                goal_state = State(-1, Location(goal[0], goal[1])) 
                agent_dict[name] = {'start':start_state, 'goal':goal_state}
            else:
                raise Exception(f"INFEASIBLE AGENT LOCATION")

        return agent_dict

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)
    
    def get_visit_dict(self, solution_df):
        used_states = solution_df.values.flatten()
        return {v.location: sum(map(v.is_equal_except_time, used_states)) for v in self.grid_locations}
        
    def get_neighbors(self, state, agent):
        neighbors = []
        for action in self.actions:
            new_state = state.move(1, action)
            if self.state_valid(new_state, agent) and self.transition_valid(state, new_state, agent)\
                and self.check_other_goals(new_state):
                neighbors.append(new_state)
        return neighbors
    
    def check_other_goals(self, state):
        check_list = list(map(state.is_equal_except_time, self.current_reached_goals))
        if any(check_list):
            for goal in np.array(self.current_reached_goals)[check_list]:
                if state.time >= goal.time:
                    return False
        return True
    
    def check_goal(self, goal):
       
        if len(self.current_tabu) > 0:
            check_lst = list(map(goal.is_equal_except_time, self.current_tabu))
            if any(check_lst):
                return np.array(self.current_tabu)[[check_lst]][0].time #False
        
        max_len = self.current_solution_df.shape[0]
        if max_len > goal.time:    
            for t in range(goal.time, max_len):
                constraints = self.current_constraints.setdefault(t, [])
                if any(map(goal.is_equal_except_time, constraints)):
                    return t #False
        return -1 #True
    
    def state_valid(self, state, agent):      
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and (state.location.x, state.location.y) not in self.obstacles \
            and state not in self.current_constraints.setdefault(state.time, []) \
            and Constraint(state.time, state.location) not in self.current_tabu
     
    def transition_valid(self, state_1, state_2, agent):
        return (not (State(state_1.time, state_2.location) in self.current_constraints.setdefault(state_1.time, []) \
           and State(state_2.time, state_1.location) in self.current_constraints.setdefault(state_2.time, [])) ) \
           and Constraint(state_1.time, state_1.location, state_2.location) not in self.current_tabu 
    
    def plan_path(self, agent, node, to_be_ignored=[], ignore_all=False, find_all_opt=False):
        self.current_solution_df = node.df
        if ignore_all or node.df.shape[0] == 0:
            self.current_reached_goals = []
            self.current_constraints = {}
        else:
            self.current_constraints = dict(enumerate(node.df.drop(to_be_ignored, axis=1).values.tolist()))
            self.current_reached_goals = [node.goal_states[agent] for agent in node.agents if agent not in to_be_ignored]
        
        a_star = AStar(self, agent)
        local_solution = a_star.search(find_all_opt=find_all_opt)
        
        return local_solution
    
    def replan_path(self, agent, node, not_ignored):
        self.current_solution_df = node.df
        self.current_constraints = dict(enumerate(node.df[list(not_ignored)].values.tolist()))
        node.check_tabu(agent)
        self.current_tabu = node.tabu_states[agent]
        self.current_reached_goals = [node.goal_states[_agent] for _agent in not_ignored]
        a_star = AStar(self, agent)
        local_solution = a_star.search()     
        return local_solution
        
    def partial_plan(self, agent, node, old_path, from_, to_, not_ignored):
        self.current_solution_df = node.df
        self.current_constraints = dict(enumerate(node.df[list(not_ignored)].values.tolist()))
        self.current_tabu = node.tabu_states[agent]
        self.current_reached_goals = [node.goal_states[_agent] for _agent in not_ignored]

        if to_.time == node.goal_states[agent].time:
            end_at_goal=True
        else:
            end_at_goal=False
        a_star = AStar(self, agent, end_at_goal=end_at_goal)
        partial_path = a_star.search(from_, to_)

        if partial_path:
            if partial_path[-1].time == old_path[to_.time].time:
                local_solution = old_path[:from_.time] + partial_path + old_path[to_.time+1:]
            else:
                local_solution = old_path[:from_.time] + partial_path
            return local_solution
        else:
            return False
    
    def generate_node_id(self, constraint_set):
        node_id = ""
        for agent, constraints in constraint_set.items():
            node_id += self.agent_codes[agent] + "-" + constraints.code
        return hash(node_id)
    

class SolutionNode(object):
    def __init__(self, agents): 
        self.agents = agents
        self.n_agents = len(agents)
        self.pairs = list(combinations(self.agents, 2))
        self.dict = {} # agent-path_no pairs 
        self.df = pd.DataFrame(columns=agents) # stores states of agents each time step until the longest path ends
        self.goal_states = {a:np.nan for a in agents}
        self.waiting_at_goal = {a:0 for a in agents} # each agent's waiting time at its goal state
        self.max_path_len = 0 
        self.cost = 0 # sum of path lengths
        self.DM = pd.DataFrame(columns=agents, index=agents).fillna(0)
        self.CT = pd.DataFrame(columns=agents, index=agents).fillna(0)
        self.n_conflicts = 0 
        self.conflicts = dict()
        self.dependencies = {a:set() for a in agents}
        self.tabu_states = {a:[] for a in agents}
        self.tabu_tenure = {a:0 for a in agents}
    
    def __hash__(self):
        return self.id
    def __eq__(self, other):
        return self.cost == other.cost and self.id == other.id
    def __lt__(self, other):
        return self.cost < other.cost
    def __getitem__(self, agent):
        return self.df[agent]
    
    def get_path(self, agent):
        global paths_dict
        return paths_dict[self.dict[agent]]
    
    def get_solution_dict(self):
        global paths_dict
        return {agent: paths_dict[path_no] for agent, path_no in self.dict.items()}
    
    def cost_of_paths(self):
        global paths_dict
        return sum([len(paths_dict[path_no])-1 for path_no in self.dict.values()])
    
    def update(self, agent=None, path=None, all_agents=False, count_all_conflicts=False):
        global paths_dict
        if all_agents:
            self.update_df(all_agents=all_agents)
        else:
            self.goal_states[agent] = path[-1]
            path_no = paths_dict.add(agent, path)
            self.dict.update({agent:path_no})
            self.update_df(agent, path)
        
        self.cost = self.cost_of_paths()
        if count_all_conflicts:
            self.count_conflicts(all_=True)
        else:
            self.n_conflicts = self.count_vertex_conflicts()

    def update_df(self, agent=None, path=None, all_agents=False):
        global paths_dict
        
        if all_agents:
            self.max_path_len = max([len(paths_dict[path_no]) for path_no in self.dict.values() ])
            self.df = pd.DataFrame(columns=self.agents, index = np.arange(self.max_path_len))
            for _agent, path_no in self.dict.items():
                path = paths_dict[path_no]
                self.goal_states[_agent] = path[-1]
                self.waiting_at_goal[_agent] = self.max_path_len - len(path)
                self.df[_agent] = path + [path[-1].move(t) for t in range(1, self.waiting_at_goal[_agent]+1)] 

        else:
            new_path_len = len(path)
            len_diff = new_path_len - self.max_path_len
            if len_diff > 0:
                self.max_path_len = new_path_len
                larger_df = pd.DataFrame(columns=self.agents, index=range(self.max_path_len)) 
                larger_df.loc[self.df.index] = self.df
                self.df = larger_df
                
                for _agent, path_no in self.dict.items():
                    path = paths_dict[path_no]
                    self.goal_states[_agent] = path[-1]
                    self.waiting_at_goal[_agent] = self.max_path_len - len(path)
                    self.df[_agent] = path + [path[-1].move(t)   for t in range(1, self.waiting_at_goal[_agent]+1)] 
            else:
                self.goal_states[agent] = path[-1]
                self.waiting_at_goal[agent] = -len_diff
                self.df[agent] = path + [path[-1].move(t) for t in range(1, self.waiting_at_goal[agent]+1)]
   
    def count_conflicts(self, agents=[], all_=False):
        self.n_conflicts = self.count_vertex_conflicts() + self.count_edge_conflicts(agents, all_)
        return self.n_conflicts
    
    def count_vertex_conflicts(self):
        return self.df.size - sum(self.df.nunique(axis=1))
    
    def count_edge_conflicts(self, agents=[], all_=False):
        n_conflicts = 0
        others = list(set(self.agents) - set(agents))
        
        if all_:
            agents = self.agents
        
        elif not isinstance(agents, (list, set)):
            agents= [agents]
     
        for a in agents:
            if all_:
                others = list(set(self.agents) - set([a]))
            for t in range(self.df.shape[0]-1):
                
                filt = list(map(self.df.loc[t, a].is_equal_except_time, self.df.loc[t+1, others]))
                
                n_conflicts += sum(list(map(self.df.loc[t+1, a].is_equal_except_time, self.df[others].loc[t, filt])))
        
        return n_conflicts // 2

    def count_conflicts_in_edge(self, state_1, state_2, compared_df):
        filt = list(map(state_1.is_equal_except_time, compared_df.loc[state_1.time+1]))
        n_conflicts = sum(list(map(state_2.is_equal_except_time, compared_df.loc[state_1.time, filt])))
        return n_conflicts

    def construct_conflict_table(self):
 
        tabu = {a:[] for a in self.agents}
        self.CT.loc[:,:] = 0
     
        for agent_1, agent_2 in self.pairs:
 
            for t in range(self.max_path_len-1):
                state_1a, state_1b = self.df.loc[t:t+1, agent_1]
                state_2a, state_2b = self.df.loc[t:t+1, agent_2]
                
                if state_1a.location == state_2a.location:
                    self.CT.loc[agent_1, agent_2] += 1
                    ct = Constraint(state_1a.time, state_1a.location)
                    tabu[agent_1].append(ct)
                    tabu[agent_2].append(ct)
                    
                if state_1a.location == state_2b.location and state_1b.location == state_2a.location: 
                    self.CT.loc[agent_1, agent_2] += 1    
                    ct1 = Constraint(state_1a.time, state_1a.location, state_1b.location)
                    ct2 = Constraint(state_2a.time, state_2a.location, state_2b.location)
                    tabu[agent_1].append(ct1)
                    tabu[agent_2].append(ct2)

            self.CT.loc[agent_2, agent_1] = self.CT.loc[agent_1, agent_2]
        
        self.n_conflicts = self.CT.sum().sum() // 2
        
        return self.CT, tabu
    
    def update_conflict_table(self, replanned_agents):
        tabu = {a:[] for a in self.agents}
        
        other_agents = set(self.agents) - set(replanned_agents) #replanned in a ordered fashion
        for agent_1 in replanned_agents:
            for agent_2 in other_agents:
                self.CT.loc[agent_1, agent_2] = 0
                for t in range(self.max_path_len-1):
                    state_1a, state_1b = self.df.loc[t:t+1, agent_1]
                    state_2a, state_2b = self.df.loc[t:t+1, agent_2]

                    if state_1a.location == state_2a.location:
                        self.CT.loc[agent_1, agent_2] += 1
                        ct = Constraint(state_1a.time, state_1a.location)
                        tabu[agent_1].append(ct)
                        tabu[agent_2].append(ct)

                    if state_1a.location == state_2b.location and state_1b.location == state_2a.location: 
                        self.CT.loc[agent_1, agent_2] += 1    
                        ct1 = Constraint(state_1a.time, state_1a.location, state_1b.location)
                        ct2 = Constraint(state_2a.time, state_2a.location, state_2b.location)
                        tabu[agent_1].append(ct1)
                        tabu[agent_2].append(ct2)

                self.CT.loc[agent_2, agent_1] = self.CT.loc[agent_1, agent_2]
        
        self.n_conflicts = self.CT.sum().sum() // 2
        
        return self.CT, tabu
    
    def construct_dissimilarity_matrix(self, gamma):
        for agent_1, agent_2 in self.pairs:
            diss = path_dissimilarity(self.get_path(agent_1), self.get_path(agent_2), gamma)
            self.DM.loc[agent_1, agent_2] = diss
            self.DM.loc[agent_2, agent_1] = diss
        return self.DM

    def get_constraints(self, agent, other_agents=[]):
        if not isinstance(other_agents, (list, set)):
            other_agents = list(other_agents)
        elif other_agents==[]:
            other_agents = set(self.agents) - {agent}
            
        constraints = {}#[]
        for other in other_agents:
            for t in range(self.max_path_len-1):
                
                state_1a, state_1b = self.df.loc[t:t+1, agent]
                state_2a, state_2b = self.df.loc[t:t+1, other]

                if state_1a.location == state_2a.location:
                    constraints[t] = Constraint(state_1a.time, state_1a.location)

                elif state_1a.location == state_2b.location and state_1b.location == state_2a.location:
                    constraints[t] = Constraint(state_1a.time, state_1a.location, state_1b.location)
    

        return [ct for t, ct in sorted(constraints.items())]
    
    def check_tabu(self,agent):
        others_df = self.df.drop(agent, axis=1)
        new_tabu = deepcopy(self.tabu_states[agent])
        for ct in self.tabu_states[agent]:
            if ct not in others_df.loc[ct.time].values.tolist():
                new_tabu.remove(ct)
        self.tabu_states[agent] = new_tabu    

class Solver(object):
    def __init__(self, environment, params):
        self.env = environment
        self.infeasible_set = []
        self.feasible_set = []
        self.start_T = params['start_T']
        self.end_T = params['end_T']
        self.cooling = params['cooling']
        self.tenure = params['tabu_tenure']
        self.max_init_iter = params['max_init_iter']
        self.time_limit = params['time_limit']
        self.gamma = params['gamma']
        self.T = self.start_T
        self.last_T_incumbent_update = self.start_T
        self.n_iter = 1
        self.st = 0
        
    def search(self):
        global paths_dict
        
        self.st = time()
        
        is_optimal, current, incumbent = self.construct_init_solutions()
       
        while (not self.termination_criteria()) and (not is_optimal):
            print(f"------- Iteraton-{self.n_iter} | Time: {round(time()-self.st,2)} | T: {round(self.T,2)} | Incumbent cost: {incumbent.cost} -------")    
            self.update_tabu_list(current)
            clusters = self.generate_clusters(current)
            random.shuffle(clusters)
            
            for i, cluster in enumerate(clusters):
                size = self.set_subset_size(cluster)
                subset = random.sample(cluster, size)
                
                print(f"Replanning a subset at size {size} from Cluster-{i+1} | Current cost: {current.cost} | #conflicts: {current.n_conflicts}")
                candidate = self.replan(current, cluster, subset)

                if candidate:
                    if candidate.n_conflicts > 0 and candidate.cost+candidate.n_conflicts < incumbent.cost:
                        candidate = self.repair(candidate)
        
                    current, incumbent, is_optimal = self.metropolis_criterion(candidate, current, incumbent)

            self.n_iter += 1
            self.T -= self.cooling
            
        execution_time = round(time()-st,2)
        print("------>> SEARCH ENDED | Incumbent solution:")
        print(f"Cost: {incumbent.cost} | #conflicts: {incumbent.n_conflicts} | SA Execution Time: {execution_time}")
                   
        return incumbent.get_solution_dict()
    
    
    def update_tabu_list(self, solution):
        for agent in solution.agents:
            solution.tabu_tenure[agent] -= 1
            solution.tabu_states[agent] = []
            
    def set_subset_size(self, cluster):
        return round(len(cluster)* (1-self.T/self.start_T)) 
    
    def replan(self, current, cluster, subset):
        candidate = deepcopy(current)
        replanned_agents = set()
        random.shuffle(subset)
        
        for agent in subset:
            if candidate.tabu_tenure[agent] <= 0:
                dependencies = candidate.CT[candidate.CT[agent] > 0].index.tolist()
                not_ignored = replanned_agents | set(dependencies)
                path = self.env.replan_path(agent, candidate, not_ignored)
                replanned_agents.add(agent)
                if not path:
                    print("NOT FOUND PATH FOR", agent)
                    continue
                
                candidate.update(agent, path)
                candidate.tabu_tenure[agent] = self.tenure
                
        candidate.update_conflict_table(subset)
        
        return candidate
    
    def repair(self, cand):
        print(f"Repairing {cand.n_conflicts} conflicts in Candidate with cost of {cand.cost}")

        conflict_numbers = cand.CT.sum().sort_values()
        conflicting_agents = conflict_numbers[conflict_numbers > 0]
        ca = deepcopy(conflicting_agents)
        
        while len(conflicting_agents) > 0:
            print(f"Conflicting agents:\n{conflicting_agents}")
            
            agent = random.choice(conflicting_agents[conflicting_agents==max(conflicting_agents)].index.tolist())
            conflicting_agents.drop(agent, inplace=True)
            
            cand.tabu_states[agent] = cand.get_constraints(agent, cand.CT[cand.CT[agent]>0].index.tolist() )
            
            if len(cand.tabu_states[agent]) == 0 or cand.tabu_tenure[agent] > 0:
                continue
            
            first_conflict_t, last_conflict_t = cand.tabu_states[agent][0].time, cand.tabu_states[agent][-1].time
            old_path = cand.get_path(agent)
            path_len = len(old_path) - 1
            
            if path_len > 50:
                range_size = max(path_len//4, 20)
                if first_conflict_t >= path_len:
                    start_t, end_t = path_len-range_size, path_len
                elif last_conflict_t >= path_len:
                    start_t, end_t = max(first_conflict_t-range_size, 0), path_len
                else:
                    start_t, end_t = max(first_conflict_t-range_size, 0), min(last_conflict_t+range_size, path_len)

                start_state, end_state = old_path[start_t], old_path[end_t] 
                new_path = self.env.partial_plan(agent, cand, old_path,
                                             from_=start_state, to_=end_state, 
                                             not_ignored=set(cand.agents)-{agent})
            else:
                new_path = self.env.replan_path(agent, cand, not_ignored=set(cand.agents)-{agent})
            
            if new_path:
                cost_change = len(new_path)-1-path_len
                
                if cost_change <= 10 * cand.CT[agent].sum():
                    cand.update(agent, new_path)
                    cand.tabu_tenure[agent] = self.tenure
                    cand.CT[agent], cand.CT.loc[agent] = 0, 0
                    conflict_numbers = cand.CT.sum().sort_values()
                    conflicting_agents = conflict_numbers[conflict_numbers > 0]
                    print(f"{agent}'s path is repaired | Cost change: {cost_change}")
                else:
                    print(f"{agent}'s path is NOT repaired | Cost change: {cost_change}")
            else:
                print(f"-----!!! {agent}'s PATH COULD NOT BE REPAIRED !!!-----")
        
        cand.count_conflicts(agents=ca.index.tolist())
        print(f"Repaired Candidate | Cost: {cand.cost} | #conflicts: {cand.n_conflicts}")
        return cand
    
    def generate_clusters(self, solution):
        DM = solution.construct_dissimilarity_matrix(self.gamma)
  
        dist_threshold = np.mean(np.triu(DM))     
        agg = AgglomerativeClustering(linkage="average", 
                                      metric='precomputed', 
                                      distance_threshold= dist_threshold,
                                      n_clusters=None)
        agg.fit(DM)
        agents= np.array(DM.columns)
        clusters = [list(agents[agg.labels_==i]) for i in range(agg.n_clusters_)]
        print(f"Clusters with the distance threshold of {round(dist_threshold,2)}:\n{clusters}")
        return clusters

    def construct_init_solutions(self):
        
        infeasible = self.construct_infeasible()
        self.lower_bound = infeasible.cost
        if infeasible.n_conflicts == 0:
            print("--------Optimal solution-------")
            print(f"Cost: {infeasible.cost} | #conflicts: {infeasible.n_conflicts}")

            return True, infeasible, infeasible
        else:
            infeasible.construct_conflict_table()
            if infeasible.n_conflicts < infeasible.n_agents**2/100:
                feasible = self.repair(infeasible)
            else:
                feasible = self.construct_feasible(infeasible)

            if feasible.cost == self.lower_bound:
                print("--------Optimal solution-------")
                print(f"Cost: {feasible.cost} | #conflicts: {feasible.n_conflicts}")
    
                return True, feasible, feasible
        

        return False, feasible, feasible
    
    def construct_infeasible(self):
        global paths_dict
        
        st = time()
        infeasible = SolutionNode(self.env.agent_names)     
        
        for agent in self.env.agent_names:
            path = self.env.plan_path(agent, infeasible, ignore_all=True)

            if not path:
                raise Exception("INFEASIBLE INSTANCE")
            
            infeasible.update(agent, path)
        
        infeasible.count_conflicts(all_=True)
        
        execution_time = round(time()-st,2)
        print(f"Init. infeasible solution | Cost: {infeasible.cost} | #conflicts:{infeasible.n_conflicts} | Construction time(sec): {execution_time}")
        
        self.infeasible_set.append(infeasible)
        
        return infeasible
    
    def construct_feasible(self, other_solution):
        global seed_number
        st = time()
        
        path_len = {agent: len(path) for agent,path in other_solution.get_solution_dict().items()}
        agents = sorted(self.env.agent_names, key=path_len.get)
        
        n_iter = 0
        while n_iter < self.max_init_iter:
            n_iter += 1
            is_feasible = True
            ignore_lst = agents.copy()
            feasible = SolutionNode(agents)

            for i,agent in enumerate(agents):
                path = self.env.plan_path(agent, feasible, to_be_ignored=ignore_lst)
                if not path:
                    print(f"NO INIT. LOCAL SOLUTION FOR {agent}")
                    print("RESTARTING FEASIBLE SOLUTION CONSTRUCTION...")
                    agents.remove(agent)
                    agents.insert(0, agent)
                    is_feasible = False
                    break
                feasible.update(agent, path)
                ignore_lst.remove(agent)
                
            if is_feasible:
                execution_time = round(time()-st,2)
                self.feasible_set.append(feasible)
                print(f"Init. feasible solution | Cost:{feasible.cost} | #conflicts: {feasible.n_conflicts} | Construction time(sec): {execution_time}")

                return feasible
            
        raise Exception("Feasible solution could not be found")
        
    def metropolis_criterion(self, candidate, current, incumbent):
        if (candidate.cost <= current.cost) | (current.n_conflicts > 0 and candidate.n_conflicts==0) | (current.n_conflicts > candidate.n_conflicts):
            current = deepcopy(candidate)

            if (current.cost < incumbent.cost and current.n_conflicts == 0)\
                or (current.cost == incumbent.cost and current.n_conflicts < incumbent.n_conflicts):
                incumbent = deepcopy(current)
               
                self.last_T_incumbent_update = self.T
                print(f"----- Better solution found with cost of {incumbent.cost} -----")
                if incumbent.cost == self.lower_bound and incumbent.n_conflicts == 0:
                    print("------Optimal solution found-------")
                    print(f"Cost: {incumbent.cost} | #conflicts: {incumbent.n_conflicts}")
                    return current, incumbent, True

        else:
            delta = candidate.cost - current.cost
            accept_prob = exp(-delta/self.T)
            if accept_prob > random.random():
                current = deepcopy(candidate)
            else:
                print(f"----- Rejected candidate with cost {candidate.cost} -----")   

        return current, incumbent, False
    
    def termination_criteria(self):
        return (self.T <= self.end_T) or (time()-self.st >= self.time_limit)
                

def main(input_dict, SA_params):
    
    file_path = "".join(list(input_dict.values()))
    if input_dict['file_type']=='.json':
        with open(file_path) as f:
            param = json.load(f) 
    else:
        with open(file_path) as param_file:
            try:
                param = yaml.load(param_file, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

    dimensions = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']

    env = Environment(dimensions, agents, obstacles)
    meta = Solver(env, SA_params)
    feasible_sol = meta.search()
    feasible_plan = generate_plan(feasible_sol)
    
    if not feasible_plan:
        print("----------Solution not found-----------" ) 
        return {}, {}
        
    return env, feasible_plan

from argparse import ArgumentParser
if __name__ == '__main__':

    parser = ArgumentParser()
    
    parser.add_argument('-k', '--n_agents', type=int, default=25, choices=[25,35,50])
    parser.add_argument('-i', '--instance_no', type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument('-t', '--time_limit', default=60)
    parser.add_argument('-n', '--n_runs', type=int, default=1)

    args = parser.parse_args()
    n_agents = args.n_agents
    instance_no = args.instance_no
    time_limit = args.time_limit
    n_runs = args.n_runs

    map_name, file_type = "random32", ".yaml"
    instance_name = f'{map_name}-{n_agents}-{instance_no}'
    input_dict = {"path": "Instances/"+map_name+"/",
                    "name": instance_name,
                    "file_type": file_type
                    }
        
    params = {'start_T': 50, 'end_T':0, 
              'cooling':5, 'tabu_tenure':4, 'time_limit':90,
              'max_init_iter':n_agents,
              'gamma':15}

    progressions = {'incumbent':[], 'current':[]}

    for sn in range(n_runs):
        random.seed(sn)
        np.random.seed(sn)

        st = time()
        print("------------------>>> NEW SEARCH STARTED <<<-----------------")
        env, feasible_plan =  main(input_dict, params)
        
        total_elapsed_time  = round(time()-st, 2)
        print("Total elapsed time:", total_elapsed_time)