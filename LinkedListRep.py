# back end for maintaining opjects that will represent the nodes
# and arrows on the "whiteboard"
import numpy as np

class node:
    def __init__(self, data):
        self.datum = data # contains the data
        self.nodes = None # contains the reference to the next node
    def add_node(self,node)
        np.insert(self.nodes,0,node)

class linked_list:
    def __init__(self):
        self.cur_node = None

    def add_node(self, data):
        new_node = node() # create a new node
        new_node.data = data
        new_node.next = self.cur_node # link the new node to the 'previous' node.
        self.cur_node = new_node #  set the current node to the new one.
        
    def get_Datum(self):
        return self.data

    def list_print(self):
        node = self.cur_node # cant point to ll!
        while node:
            print node.data
            node = node.next

   


    # def set_Datum():
