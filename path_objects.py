from functools import reduce

class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
        self.hash_value = hash(str(self.x)+"-"+str(self.y))
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return self.hash_value
    
    def __str__(self):
        return str((self.x, self.y))
    
    def __add__(self, other):
        return Location(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Location(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):
        return Location(self.x * other.x, self.y * other.y)
    
class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location
        self.hash_value = hash( str(self.time)+"-"+str(self.location.x)+"-"+str(self.location.y))                          
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location if isinstance(other, (State, Constraint)) else False 
    def is_equal_except_time(self, other):
        return self.location == other.location if isinstance(other, self.__class__) else False    
    def move(self, t=0, action=Location(0,0)):
        return State(self.time + t, self.location + action)
    def __hash__(self):
        return self.hash_value
    def __str__(self):
        return str((self.time, (self.location.x, self.location.y)))
    def __sub__(self, other):
        return State(self.time - other.time, self.location - other.location)
    def __add__(self, other):
        return State(self.time + other.time, self.location + other.location)
    def __mul__(self, other):
        return State(self.time * other.time, self.location * other.location)


class Conflict(object):
    VERTEX = 1
    EDGE = 2        
    def __init__(self, agent_1, agent_2, location_1, location_2, time, conflict_type):
        self.time = time
        self.type = conflict_type
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.location_1 = location_1
        self.location_2 = location_2

    def __str__(self):
        return f'({self.agent_1}-{self.agent_2}, ({self.time}, ({self.location_1},{self.location_2})))'
    
class Constraint(object):
    def __init__(self, time, *locations):
        self.time = time
        self.locations = list(locations)
        self.hash_value = hash(str(self.time)+reduce(lambda loc1, loc2: str(loc1)+str(loc2), self.locations, ""))  
    def __eq__(self, other):
        if isinstance(other, State):
            return False if len(self.locations) > 1 else self.time == other.time and self.locations[0] == other.location
        elif isinstance(other, Constraint):
            return self.time == other.time and self.locations == other.locations
        else:
            return False
            
    def __hash__(self):
        return self.hash_value,
    
    def __str__(self):
        return f"{str(self.time)}, {[str(loc) for loc in self.locations]}" 
    
class ConstraintsDict(object): #CURRENTLY NOT USED
    #a single agent dictionary of avoided agent-constraints pairs
    #keeps the information of which constraint is applied to avoid from a conflict with which agent
    
    def __init__(self):
        self.constraints = {}
        self.code = ""

    def add_constraint(self, new_constraint):
        global constraint_code
        for avoided_agent in new_constraint:
            self.constraints.setdefault(avoided_agent, []).extend(new_constraint[avoided_agent])
        self.code = "-".join(sorted([ constraint_code[ct] for ct in self.values()]))
         
    def values(self):
        return [ct for lst in self.constraints.values() for ct in lst]
    
    def pop(self, avoided_agent):
        try:
            self.constraints.pop(avoided_agent)
        except:
            pass
    
    def __len__(self):
        return len(self.constraints)
    
    def __str__(self):
        return str([str(ct) for ct in self.constraints.values()])
    
class PathsDict(object): 
    # stores all found paths
    # aimed to not search a path again, which is found in previous iterations
    # keeps the information of found paths under which constraints for each agent 
    
    def __init__(self, agents):
        self.paths = {} # path_no-path pairs, each new code is current length of paths dict
        self.len = 0    # each path is a list of states ordered in time
        self.pdict = {a:{} for a in agents} #nested dictionary of agent-constraints-path
    
    def __getitem__(self, path_no):
        return self.paths[path_no]
    
    def __keys__(self):
        return self.paths.keys()
    
    def __values__(self):
        return self.paths.values()
    
    def __len__(self):
        return len(self.paths)
    
    def items(self):
        return self.paths.items()
    
    def get(self, agent, ct_code):
        return self.pdict[agent][ct_code]
    
    def add(self, agent, path, ct_code=""):
        self.len += 1
        self.paths[self.len] = path
        self.pdict[agent][ct_code] =  self.len
        return self.len # path_no
            