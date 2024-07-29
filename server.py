import tensorflow as tf
import numpy as np

class Server:
    def __init__(self):     
        self.generated_data=None
      
    def distribute_between_edgeservers(self,edgeservers):       
        #self.generated_data=self.generated_data.shuffle(self.generated_data.cardinality(),reshuffle_each_iteration=False)
        data_list=[]
        label_list=[]
        for i,j in self.generated_data:
            data_list.append(i)
            label_list.append(j)
        data_list=np.array(data_list)
        label_list=np.array(label_list)
        gen_data_idxs=[tf.argmax(label_list[i]) for i in range(len(label_list))]
        for label in range(10):
            g_idx=[]
            times=0
            edges_list=[]
            for j,d in enumerate(gen_data_idxs):
                if label==d:
                    g_idx.append(j)
            np.random.shuffle(g_idx)
            for edge in edgeservers:
                index=int(edge.name.split('_')[1])-1 
                if label in edge.classes:
                    edges_list.append(index)
                    times+=1
            gen_split=np.array_split(g_idx,times) 
            j=0
            for index in edges_list:
                if edgeservers[index].generated_data:
                    edgeservers[index].generated_data=edgeservers[index].generated_data.concatenate(
                             tf.data.Dataset.from_tensor_slices((data_list[gen_split[j]],label_list[gen_split[j]])))
                    j+=1
                else:
                    edgeservers[index].generated_data=tf.data.Dataset.from_tensor_slices((data_list[gen_split[j]],
                                                                                          label_list[gen_split[j]])) 
                    j+=1
                  
    def refresh_server(self):                   
        self.generated_data=None
