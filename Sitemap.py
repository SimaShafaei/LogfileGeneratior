# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:45:39 2023

@author: sshaf
"""

import random
import numpy as np
import string
import pandas as pd
from scipy.stats import truncnorm

class SiteMap:
    def __init__(self):
        self.structure={}
        self.Tasklist=[]
    
    def get_task_list(self):
        return self.Tasklist
        
 
    #Create a random name for a page created by a combination of alphabet and digits
    def create_page_names(self,pageNum):
        page_names = []
        for i in range(pageNum):
            page_names.append(''.join(random.choices(string.ascii_letters + string.digits, k=10)))
        return page_names
    
    #Get the path of reaching the page from the root
    def get_page_path(self, page_name):
        path = ""
        queue = [(self.structure, "")]
        while queue:
            current_node, current_path = queue.pop(0)
            for key, value in current_node.items():
                if key == page_name:
                    path = current_path + "/" + key
                    break
                elif isinstance(value, dict):
                    queue.append((value, current_path + "/" + key))
        
        path_parts = [part for part in path.split('/') if part]
        return path_parts
    
    #Add a single page under the determined parent node       
    def _add_page(self, parent_page, new_page):
        if not parent_page:
            self.structure[new_page] = {}
        else:
            pages_to_search = [self.structure]
            while pages_to_search:
                page = pages_to_search.pop()
                for key, value in page.items():
                    if key == parent_page:
                        value[new_page] = {}
                        return
                    elif value:
                        pages_to_search.append(value)
            raise ValueError("Parent node not found in the tree.")
    
    
    #Add a new page list as the child of an existing page
    def add_page_list(self, parent_page, new_pages):
        for page in new_pages:
            self._add_page(parent_page, page)
            parent_page=page
        path=self.get_page_path(parent_page)
        self.Tasklist.append(path)
    
    # Get depth of the SiteMap 
    def depth(self):
        return self._depth_helper(self.structure)
    
    def _depth_helper(self,tree):
        if not tree:
            return 0
        max_depth = 0
        for child in tree:
            depth =self._depth_helper(tree[child])
            if depth > max_depth:
                max_depth = depth
        return max_depth + 1
        
    

    #WorkHorse for extracing the name of all pages in the sitemap at depth k
    def _get_pages_at_depth_helper(self,tree,k):
        if k == 1:
            return list(tree.keys())  # return root node at depth 0

        pages_at_depth_k = []
        for key in tree:
            if isinstance(tree[key], dict):
                pages_at_depth_k += self._get_pages_at_depth_helper(tree[key], k-1)
        return pages_at_depth_k
    
    #Driver for extracing the name of all pages in the sitemap at depth k
    def get_pages_at_depth(self,k):
        d=self.depth()
        if k<=d:
            return self._get_pages_at_depth_helper(self.structure,k)
        else:
            return []
        
        
    #WorkHorse for extracing the name of all pages in the sitemap
    def _get_all_pages_helper(self,tree):
        pages = list(tree.keys())  # add root node to list of nodes
        for key in tree:
            if isinstance(tree[key], dict):
                pages+=self._get_all_pages_helper(tree[key])
        return pages

    #Driver for extracing the name of all pages in the sitemap
    def get_all_pages(self):
        return self._get_all_pages_helper(self.structure)
    
    #WorkHorse for printing the sitemap
    def __str__(self):
        return self._str_helper(self.structure, 0)

    def _str_helper(self, node, depth):
        result = ''
        indent = '  ' * (depth-1)+'|__'
        for key in node:
            result += f'{indent}{key}\n'
            if isinstance(node[key], dict):
                result += self._str_helper(node[key], depth+2)
        return result
    
    #WorkHorse for determining number of pages in the sitemap
    def _get_num_pages_helper(self,tree):
        num_pages = 0
        for key in tree:
            if isinstance(tree[key], dict):
                num_pages += self._get_num_pages_helper(tree[key])+1
            else:
                num_pages += 1
        return num_pages
    
     #Driver for determining number of pages in the sitemap
    def get_num_pages(self):
        return self._get_num_pages_helper(self.structure)
    
    
    
    def generate_random_sitemap(self,n_task,mu_pages,sd_pages):
        # Determine number of pages per task
        #num_pages = np.round(np.random.normal(mu_pages, sd_pages, n_task)).astype(int)
        
        # Generate random numbers of page from a truncated normal distribution
        lower_bound = (2 - mu_pages) / sd_pages
        random_numbers = truncnorm.rvs(lower_bound, np.inf, loc=mu_pages, scale=sd_pages, size=n_task)
        # Convert the generated numbers to integers
        num_pages = np.ceil(random_numbers).astype(int)
        
        #df = pd.DataFrame(columns=['ID', 'Num_Pages', 'Page_Names'])
        for task, pageNumber in enumerate(num_pages):
            # Create and add first task to the sitemap
            if task == 0: 
                pageNames=self.create_page_names(pageNumber)
                self.structure={}
                current_node = self.structure
                for page in pageNames:
                    current_node[page] = {}
                    current_node = current_node[page]
                self.Tasklist.append(pageNames)
                    
            # Create and add other tasks to the sitemap
            else:
                dep=self.depth()
                k = random.randint(1, min(pageNumber-1,dep))
                potential_parents=self.get_pages_at_depth(k)
                new_task_pages=self.create_page_names(pageNumber-k)
                parent = random.choice(potential_parents)
                self.add_page_list(parent,new_task_pages)
                

            # Save task id and its pagename in a dataframe
            #new_row = {'ID': task, 'Num_Pages': pageNumber, 'Page_Names': pageNames}
            #df = df.append(new_row, ignore_index=True)
        return self.structure