class Graph:
    def __init__(self, name, v, e, tri_cnt, max_degree, mem_look_up_time):
        self.name = name
        self.v = v
        self.e = e
        self.tri_cnt = tri_cnt
        self.max_degree = max_degree
        self.mem_look_up_time = mem_look_up_time


class Pattern:
    def __init__(self, in_type, k):
        self.type = in_type
        self.k = k


graphs = {}
patterns = {"triangle": Pattern("clique", 3), "4clique": Pattern("clique", 4)}
