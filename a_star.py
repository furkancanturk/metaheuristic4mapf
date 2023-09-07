from itertools import combinations, product
from copy import deepcopy
from time import time 
from math import fabs
import pandas as pd
import random
INF = float("inf")


def print_obj_lst(lst):
    print([str(obj) for obj in lst])
    
class AStar():
    def __init__(self, env, agent_name, end_at_goal=True):
        self.n_iter = 0 
        self.total_nodes = 0     
        self.agent_name = agent_name
        self.start_state = env.agent_dict[agent_name]["start"]
        self.goal_state = env.agent_dict[agent_name]["goal"]
        self.end_at_goal = end_at_goal
        
        self.get_neighbors = env.get_neighbors
        self.check_goal = env.check_goal
        
        self.CAT = env.current_solution_df.drop(agent_name, axis=1)
        self.old_path = env.current_solution_df[agent_name]

        self.closed_set, self.open_set = set(), set()
        self.f_score, self.g_score, self.h_score, self.num_conflict, self.came_from = {}, {}, {}, {}, {}
        self.step_cost = 1

    def admissible_heuristic(self, state):
        return fabs(state.location.x - self.goal_state.location.x) + fabs(state.location.y - self.goal_state.location.y)
    
    def reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]
    
    def get_best_state(self):
        f_benchmark = {state: self.f_score[state] for state in self.open_set}
        min_f = min(f_benchmark.values())
        min_f_states = [state for state in self.open_set if self.f_score[state] == min_f]
        for state in min_f_states:
            if self.end_at_goal:
                if state.is_equal_except_time(self.goal_state):
                    #return True, state
                    conflict_t = self.check_goal(state)
                    if int(conflict_t) == -1:
                        return True, state
                    else:
                        current = self.wait(conflict_t, state)
                        return False, current
            else:
                if state.is_equal_except_time(self.goal_state):
                    if state.time == self.goal_state.time:
                        return True, state
                    else:
                        self.goal_state = self.old_path.iloc[-1]
                        self.end_at_goal = True

        if len(min_f_states) > 1:
            conflict_benchmark = {state: self.num_conflict[state] for state in min_f_states}                            
            min_conflict = min(conflict_benchmark.values())
            min_conflict_states = [state for state, n_conflict in conflict_benchmark.items() if n_conflict == min_conflict]
            
            if len(min_conflict_states) > 1:
                h_benchmark = {state: self.h_score[state] for state in min_conflict_states}
                min_h = min(h_benchmark.values())
                min_h_states = [state for state, h in h_benchmark.items() if h == min_h]
                
                random.shuffle(min_h_states)
                current = min_h_states[0]
                if len(min_h_states) > 1 and self.old_path.shape[0] > current.time:
                    for state in min_h_states:
                        if state != self.old_path[state.time]:
                            current = state
                            break
            else:
                current = min_conflict_states[0]
        else:
            current = min_f_states[0]
            
        return False, current
    
    def wait(self, t, state):
        conflicted_state = deepcopy(state)
        total_time_diff = t-state.time
        for time_diff in range(total_time_diff-1):
            next_state = state.move(1)
            self.came_from[next_state] = state
            state = next_state

        self.num_conflict[state] = self.num_conflict[conflicted_state]
        self.g_score[state] = self.g_score[conflicted_state]
        self.h_score[state] = 0
        self.open_set.remove(conflicted_state)
        self.closed_set.add(conflicted_state)
        self.open_set.add(state)
   
        return state
                        
    def count_conflicts(self, current_state, next_state):
        n_vertex_conflicts = sum(self.CAT.loc[next_state.time] == next_state)
        filt = list(map(current_state.is_equal_except_time, self.CAT.loc[next_state.time]))
        n_edge_conflicts = sum(list(map(next_state.is_equal_except_time, self.CAT.loc[current_state.time, filt])))
        return n_vertex_conflicts + n_edge_conflicts  

    def search(self, from_=None, to_=None, find_all_opt=False):
        
        self.start_state = from_ if from_ else self.start_state
        self.goal_state = to_ if to_ else self.goal_state
        self.open_set.add(self.start_state)
        self.num_conflict[self.start_state] = 0
        self.g_score[self.start_state] = 0
        self.h_score[self.start_state] = self.admissible_heuristic(self.start_state)
        self.f_score[self.start_state] = self.h_score[self.start_state] 
        
        while self.open_set:
            self.n_iter += 1
     
            is_goal, current = self.get_best_state()

            if is_goal:
                if find_all_opt:
                    return self.opt_search(agent_name, f_score[current], min_f_states)
                else:
                    return self.reconstruct_path(current)

            if current.time > max(self.h_score[self.start_state] * 2, 100):
                print("Too long path for", self.agent_name)
                break

            neighbor_list = self.get_neighbors(current, self.agent_name)
       
            for neighbor in neighbor_list:
                tentative_g_score = self.g_score.setdefault(current, INF) + self.step_cost

                if neighbor in self.open_set and tentative_g_score >= self.g_score.setdefault(neighbor, INF):
                    continue
                elif neighbor in self.closed_set:
                    if  tentative_g_score >= self.g_score.setdefault(neighbor, INF):
                        continue
                    else:
                        self.closed_set.remove(neighbor)

                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.h_score[neighbor] = self.admissible_heuristic(neighbor)
                self.f_score[neighbor] = self.g_score[neighbor] + self.h_score[neighbor]
        
                if self.CAT.shape[0] == 0:
                    self.num_conflict[neighbor] = 0
                elif neighbor.time >= self.CAT.shape[0]:
                    self.num_conflict[neighbor] = self.num_conflict[current]
                else:
                    self.num_conflict[neighbor] = self.count_conflicts(current, neighbor) + self.num_conflict[current]
                
                self.open_set |= {neighbor}
                self.total_nodes += 1
                
            self.open_set.remove(current)
            self.closed_set.add(current)
 
        return False
    
    def opt_search(self, agent_name, opt_cost, candidate_states):

        goal_state = self.agent_dict[agent_name]["goal"]
        closed_set = set()
        open_set = candidate_states
        opt_paths = []
        while len(open_set) > 0:
            
            copy_open = deepcopy(open_set)
            for state in copy_open:
                if state.is_equal_except_time(goal_state):
                    opt_paths += [self.reconstruct_path(state)]
                    open_set.remove(state)
            if len(open_set) == 0:
                break
            
            current = open_set.pop()
            closed_set.add(current)
            neighbor_list = self.get_neighbors(current, agent_name)
           
            for neighbor in neighbor_list:
                tentative_g_score = self.g_score.setdefault(current, INF) + self.step_cost
                
                if neighbor in open_set and tentative_g_score >= self.g_score.setdefault(neighbor, INF):
                    continue
                elif neighbor in closed_set:
                    if  tentative_g_score >= self.g_score.setdefault(neighbor, INF):
                        continue
                    else:
                        closed_set.remove(neighbor)
                
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.h_score[neighbor] = self.admissible_heuristic(neighbor)
                self.f_score[neighbor] = self.g_score[neighbor] + self.h_score[neighbor]
                if f_score[neighbor] == opt_cost:
                    open_set.add(neighbor) 
                    self.total_nodes += 1
                else:
                    closed_set.add(neighbor)           
                        
        return opt_paths