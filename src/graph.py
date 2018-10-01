import numpy as np
import random

class Node:
    
    def __init__(self,node):
        self.id = node
        self.parent = None
        self.children = []
        self.abundance = 0.0
        self.layer = 0
    
    def add_child(self, child):
        self.children.append(child)
        
    def get_children(self):
        return self.children
    
    def get_children_ids(self):
        out = []
        for c in self.children:
            out.append(c.id)
        return out
        
    def set_parent(self, parent):
        self.parent = parent
        self.layer = parent.get_layer() + 1
        
    def get_parent(self):
        return self.parent
      
    def get_layer(self):
        return self.layer
      
    def get_id(self):
        return str(self.id)
    
    def set_id(self, id):
        self.id = id
    
    def get_abundance(self):
        return self.abundance
    
    def set_abundance(self, x):
        self.abundance = x
    
    def set_layer(self, x):
        self.layer = x
    
    def calculate_abundance(self):
        calc = 0
        for c in self.children:
            calc += c.abundance
        self.abundance = calc

    def get_leaves(self):
        calc = 0
        c = self.children
        while len(c) != 0:
            n = c.pop()
            if len(n.get_children()) == 0:
                calc = calc + 1
            else:
                for child in n.get_children(): 
                    c.append(child)
        return calc
 
class Graph:       
    def __init__(self):
        self.nodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        self.width = 0
        self.layers = 0
        self.root = None
        self.NODE_DICT = {}
	self.node_count = 0
    
    def __iter__(self):
        return iter(self.nodes.values())
    
    def add_node(self, l, node):
        self.nodes[l][node] = node
        self.NODE_DICT[str(node.get_id())] = node
    
    def get_node(self, l, n):
        if n in self.nodes[l]:
            return self.nodes[l][n]
        else:
            return None
    
    def get_node_by_name(self, n):
        for l in range(0,self.layers+1):
            for node in self.nodes[l]:
                if node.get_id() == n:
                    return self.nodes[l][node]
        return None
    
    def get_nodes(self, l):
        return self.nodes[l]
   
    def get_nodes_ids(self, l):
	nlist = self.nodes[l]
	out = []
 	for n in nlist:
		out.append(n.get_id())
	return sorted(out)
     
    def get_node_count(self):
	return self.node_count
 
    def get_dictionary(self):
        return self.NODE_DICT
    
    def build_graph(self, mapfile):
        self.node_count = 0
	my_map = []
        my_list=[]
        layer = -1
        node_stack = []
        current_node = None
        current_parent = None
        max_layer = 0      
	num_c = 0 
        with open(mapfile) as fd:
            for line in fd:
                segment = line.split(',')
                for sentence in segment:
                    drop = sentence.count("(")
                    for i in range(0,drop):
                        layer = layer + 1
                        node_stack.append(Node(str(layer)))
                    sentence = sentence.replace("(","")
                    words = sentence.split(")")
                    layer = layer+1
                    if (layer > max_layer):
                        self.layers = layer
                        max_layer = layer                       
                    for w in range(0,len(words)):
                        if (w == 0):
                            current_node = Node(words[w])
			    num_c += 1
                        else:
                            layer = layer - 1 
                            current_node = node_stack.pop()
                            current_node.set_id(words[w])
                            
                        if (len(node_stack) > 0):
                            current_parent = node_stack[-1]
                            self.add_node(layer, current_node)
			    self.node_count += 1
                            current_node.set_parent(current_parent)
                            current_parent.add_child(current_node)
                            current_node.set_layer(layer)
                            
                        else:
                            self.add_node(layer,current_node)
			    self.node_count += 1
                            current_node.set_layer(layer)
                            num_c += 1
                    layer = layer -1
        self.width = sum([s.get_leaves() for s in self.get_nodes(0)])	

    def print_graph(self):
	c = 0
        for i in range(0, self.layers+1):
            for n in self.get_nodes(i):
                c += 1
		print(n.get_id() + " " + str(n.get_abundance()))
            print("\n" + str(c))              
    
    def graph_vector(self):
        c = 0
	out = []
        for i in range(1, self.layers+2):
            for n in self.get_nodes_ids(i):
                c += 1
                out.append(self.get_node_by_name(n).get_abundance())
	return out
        
    def populate_graph(self, lab, x):      
        layer = self.layers
	tracker = {}
	for f in lab:
	    tracker[f] = False
	while layer >= 0:
            for n in self.get_nodes(layer):
                if len(n.get_children()) > 0:
                    s = 0
                    for c in n.get_children():
                        s = s + float(c.get_abundance())
                    for i in range(0, len(lab)):
                        if str(n.get_id()).replace("_"," ") == str(lab[i]):
			    tracker[str(lab[i])] = True
                            s = s + x[i]
                    n.set_abundance(s)
                else:
                    for i in range(0, len(lab)):
                        if str(n.get_id()).replace("_", " ") == str(lab[i]):
			    tracker[str(lab[i])] = True
                            n.set_abundance(x[i])
	    layer = layer - 1   
 
    def get_map(self, permute=-1):
        #m = np.zeros(((self.layers), self.width))
        m = np.repeat(-1.0, self.layers * self.width).reshape((self.layers, self.width))
	current = self.get_nodes(1)
        if permute >= 0:
            print("Permuting Tree...")
        for i in range(0, self.layers+1):
            j = 0
            temp = []
            for n in current:
                m[i][j] = n.get_abundance()
                if permute >= 0:
                    np.random.seed(permute)
                    np.random.shuffle(n.get_children())
                temp = np.append(temp, n.get_children())
                j = j+1
            current = temp 
	return m        
            
    def get_sparse_map(self, permute=-1):
        m = np.zeros(((self.layers), self.width))
        nodes = list(self.get_nodes(1))
        j = 0
        if permute >= 0:
            print("Permuting Tree...")
        while len(nodes) > 0:
            n = nodes.pop()
            m[n.get_layer()-1][j] = n.get_abundance()
            c = n.get_children()
            if len(c) > 0:
                for child in c:
                    nodes.append(child)
            else:
                j = j + 1
        return m             
   
    def get_contrast_map(self, permute=-1):
        m = np.repeat(-1, self.layers * self.width).reshape(self.layers, self.width)
        nodes = list(self.get_nodes(1))
        j = 0
        if permute >= 0:
            print("Permuting Tree...")
        while len(nodes) > 0:
            n = nodes.pop()
            m[n.get_layer()-1][j] = n.get_abundance()
            c = n.get_children()
            if len(c) > 0:
                for child in c:
                    nodes.append(child)
            else:
                j = j + 1
        return m



 
    def get_ref(self):
        width = 0
        for i in range(0,self.layers):
            w = len(self.get_nodes(i))
            if w > width:
                width = w        
        self.width = width   
        m = np.zeros(((self.layers), self.width), dtype=object)
        current = self.get_nodes(1)
        for i in range(0, self.layers):
            j = 0
            temp = []
            for n in current:
                m[i][j] = n.get_id()
                temp = np.append(temp, n.get_children())
                j = j+1
            current = temp      
        return m

    def write_table(self, path):
        fp = open(path, "w")
        for i in range(0, self.layers):
            node_list = self.get_nodes(i)
            for node in node_list:
                node_id = node.get_id()
                c = node.get_children()
                for child in c:
                    child_id = child.get_id()
                    fp.write(node_id + "\t" + child_id + "\n")

    
            
