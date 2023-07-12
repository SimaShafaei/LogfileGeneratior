"""
Created on Sat Jun 1 18:13:52 2023

@author: sima shafaei
"""

import numpy as np
import pandas as pd

class Node:
    def __init__(self,index, name, mean, std):
        self.index = index
        self.name = name
        self.lower=0
        self.upper=100
        self.prior_mean = np.copy(mean)
        self.prior_std = np.copy(std)
        self.posterior_mean=0
        self.posterior_std=0
        self.value = None
        self.parents = []
        self.children = []        

    def add_parent(self, parent):
        self.parents.append(parent)        

    def add_child(self, child):
        self.children.append(child)  


class BayesNet:
    def __init__(self,num_nodes):
        # num_nodes = number of nodes (int)
        self.num_nodes=num_nodes
        
        #dictionary of names and index of nodes
        self.names_index={}
        
        # structure = shows the structre of BN,(matrix) 
        self.structure=np.empty((num_nodes, num_nodes))
        
        #Set weights of edges
        self.weights=np.empty((num_nodes, num_nodes))
        
        # the bserved node and their values and array of tuple (node, value)
        #self.observed_nodes=[]
        
        # an array that determines the order of nodes for elimination
        self.orders=[]
        
        #set Nodes
        self.nodes=[]
        for i in range(num_nodes):
            self.nodes.append(Node(i,"", mean=[], std=[]))
    
    #Set names of all nodes
    def set_node_names(self,node_names): 
        self.names_index={}
        for i in range(self.num_nodes):
            self.nodes[i].name = node_names[i] 
            self.names_index[node_names[i]]=i
        return
    
    # Set prior destribution of all nodes
    def set_priors(self, node_prior):
        for i in range(self.num_nodes):
            self.nodes[i].prior_mean =np.copy(node_prior[i]['mean'])  
            self.nodes[i].prior_std=np.copy(node_prior[i]['std'])
            self.nodes[i].lower=node_prior[i]['lower']
            self.nodes[i].upper=node_prior[i]['upper']
        return 
    
    # Set prior destribution of node i
    # the prior of nodes,  it is an array of ([mu1,...,muk],[sgma1, ..., sigmak]) with size num_nodes for a Gaussian mixture model
    def set_prior(self,node_prior,i):
        self.nodes[i].mean =np.copy(node_prior['mean'])
        self.nodes[i].std=np.copy(node_prior['std'])
        self.nodes[i].lower=node_prior['lower']
        self.nodes[i].upper=node_prior['upper']
        return
    
    # Set values of nodes with nodeIndexs (Set all observed_nodes or evidence)
    def set_values(self, assignmens):
        for name, value in assignmens.items():
            u=self.nodes[self.names_index[name]].upper
            l=self.nodes[self.names_index[name]].lower
            if value>=l and value<=u:
                self.nodes[self.names_index[name]].value =value
            elif value<l:
                self.nodes[self.names_index[name]].value=l
            else:
                self.nodes[self.names_index[name]].value=u
        return
    
    
    def set_structure(self, struct):
        self.structure=np.copy(struct) 
        # Establish parent-child relationships based on the adjacency matrix
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if struct[i][j] == 1:
                    self.nodes[i].add_child(self.nodes[j])
                    self.nodes[j].add_parent(self.nodes[i])        
        return
    
    def add_edge(self,parent_name,child_name,weight):
        parent_index=self.names_index[parent_name]
        child_index=self.names_index[child_name]
        self.structure[parent_index][child_index]=1
        self.nodes[parent_index].add_child(self.nodes[child_index])
        self.nodes[child_index].add_parent(self.nodes[parent_index])
        self.weights[parent_index][child_index]=weight
        
    #set Parameters or weights of netword
    def set_weights(self, weights): 
        self.weights= np.copy(weights)
        return

    def set_num_node(self, n):
        self.num_node=n
        return
    
    def get_structure(self):
        return self.structure    
           
    def get_num_node(self):
        return self.num_nodes
        
    def get_weights(self):
        return self.weights
        
    def _get_topological_order(self):
        orders=[]
        nodes=[i for i in range(self.num_nodes)]
        bn_struct=np.copy(self.structure)
                
        while len(nodes)>0 : 
            
            # Determine the nodes that are in the highest order (zero columns means they have no parent)
            zero_columns = np.all(bn_struct == 0, axis=0)

            # Add nodes to orders                      
            for i in range(len(zero_columns)):
                if zero_columns[i]:
                    orders.append(nodes[i])

            # Remove their columns and rows from bn_struct
            bn_struct=bn_struct[~zero_columns][:, ~zero_columns]
           
            # Remove orderd nodes from nodes
            nodes = [node for node, zero_col in zip(nodes, zero_columns) if not zero_col]
            
        self.orders=orders    
        return orders
    
    def load_bn(self,file_path):
        # Read the CSV file
        df = pd.read_csv(file_path, index_col=0)
        
        # Extract node names and node nums
        node_names = df.index.tolist()        
        self.num_nodes=len(node_names)
        #create Nodes
        self.nodes=[]
        for i in range(self.num_nodes):
            self.nodes.append(Node(i,"", mean=[], std=[]))
        
        #Set weights of edges
        self.weights=np.empty((self.num_nodes, self.num_nodes))
        w = df.iloc[:, :-4].values
        self.set_weights(w)
        
        
        # structure = shows the structre of BN,(matrix) 
        #self.structure=np.empty((self.num_nodes, self.num_nodes))
        struct=np.where(w != 0, 1, 0)
        self.set_structure(struct)
        
        #dictionary of names and index of nodes
        self.names_index={}
        self.set_node_names(node_names)
        
        # Extract the mean and std columns
        mean = df['Mean'].values.tolist()
        std = df['Std'].values.tolist()
        lower=df['LowerBound'].values.tolist()
        upper=df['UpperBound'].values.tolist()
        for i,node in enumerate(self.nodes):
            node.prior_mean=np.copy(mean[i])
            node.prior_std=np.copy(std[i])
            node.upper=upper[i]
            node.lower=lower[i]
            
        return 
        
    
    def save_to_file(self,file_path):
        node_names=list(self.names_index.keys())
        df = pd.DataFrame(self.weights, index=node_names, columns=node_names)  
        node_means=[]
        node_stds=[]
        uppers=[]
        lowers=[]
        for node in self.nodes:
            node_means.append(node.prior_mean)
            node_stds.append(node.prior_std)
            uppers.append(node.upper)
            lowers.append(node.lower)
            
        # Add the additional features as new columns
        df['Mean'] = node_means
        df['Std'] = node_stds
        df['UpperBound'] = uppers
        df['LowerBound'] = lowers
        print(node_means)    
        df.to_csv(file_path,mode='w')
        
        
    def get_posteriors(self):
        posteriors={}
        for node in self.nodes:
            posteriors[node.name]=[node.posterior_mean,node.posterior_std]
        return posteriors
       
    def perform_inference(self,query_nodes):
        #Get topological order for eleminating variables
        self._get_topological_order()
            
        for i in self.orders:
            node=self.nodes[i] 
            
            # Update posterior for Observed Nodes or Evidences
            if node.value is not None:
                node.posterior_mean=np.array([node.value])                
                node.posterior_std=np.array([0.001])   # Set to 0.001 to avoid devision by zero
                
            
            else:
                # Get mean and std of prior distribution of current node 
                mean, precision = node.prior_mean, (1.0 / (node.prior_std ** 2))
                mean = mean* precision
                
                # Obtained posterior mean and std based on posterior destribution of parents and weights.
                sum_weights=1
                precision_sum = precision
                for i, parent in enumerate(node.parents):
                    weight = self.weights[parent.index][node.index]            
                    precision = 1.0 / (parent.posterior_std ** 2)
                    precision_sum = precision_sum + (weight ** 2) * precision
                    mean = mean + (weight ** 2) * parent.posterior_mean * precision         
                node.posterior_mean=mean/precision_sum
                node.posterior_std = np.sqrt(1 / precision_sum)
        
        result=[] 
        for n in query_nodes:
            i=self.names_index[n]
            result.append(np.random.normal(self.nodes[i].posterior_mean[0], self.nodes[i].posterior_std[0]))
        return result
