from model.initialize_model import create
import gc
import copy
from cgan import CGAN_mnist
from cgan import CGAN_cifar10
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
class Edgeserver: 
    
# 1.
  def __init__(self,id_name,cnames,dataset,image_shape,latent_dim,number_classes): # ارگومان ها    
        

        n='edgeserver'
        self.name=f'{n}_{id_name+1}'
        self.cnames=cnames          
        self.generated_data=None    # received data from server/data after cgan training/generated data by its edge generator
        self.models=[]           # cgan , ...
        if dataset=="mnist":
            for i in range(len(cnames)):
                self.models.append(CGAN_mnist(image_shape,latent_dim,number_classes))  
        else:
            for i in range(len(cnames)):
                self.models.append(CGAN_cifar10(image_shape,latent_dim,number_classes))
        self.classes=[]
        
# 2.        
    def generate_fake_data(self,client,batch_size,latent_dim):        
        noise=tf.random.normal(shape=[batch_size,latent_dim])
        Y_fake=np.random.choice(client.classes,batch_size)
        Y_fake=to_categorical(Y_fake,num_classes=10)   
        X_fake=self.models[self.cnames.index(client.name)].gen_model([noise,Y_fake],training=False)
        return X_fake,Y_fake
        
# 3.
    def train_generator(self,client,batch_size,latent_dim):
        X_cgan=tf.random.normal(shape=[batch_size, latent_dim])    # noise
        Y_cgan=np.random.choice(client.classes,batch_size)
        Y_cgan=to_categorical(Y_cgan,num_classes=10 )
        gen_loss=self.models[self.cnames.index(client.name)].cgan_model.train_on_batch([X_cgan,Y_cgan], tf.ones((batch_size,1))) 
        print('-->gen_loss=%.3f' %  (gen_loss))     # number batchs?
        
# 4.
    def update_generated_data(self,client,gen_ratio,latent_dim):    # این با اون سایز بچ ها متفاوته 🟨
        with tf.device('/CPU:0'):
            X_fake,Y_fake=self.generate_fake_data(client,int(client.len.numpy()*gen_ratio),latent_dim)
        if self.generated_data:
            self.generated_data=self.generated_data.concatenate(tf.data.Dataset.from_tensor_slices((X_fake,Y_fake))) 
        else:
            self.generated_data=tf.data.Dataset.from_tensor_slices((X_fake,Y_fake))          
# 5.                                                                       
    # after cgan training or after each round-communication
    def distribute_between_clients(self,clients): 
        #self.generated_data=self.generated_data.shuffle(self.generated_data.cardinality(),reshuffle_each_iteration=False)
        data_list=[]
        label_list=[]
        for i,j in self.generated_data:
            data_list.append(i)
            label_list.append(j)
        data_list=np.array(data_list)
        label_list=np.array(label_list)
        gen_data_idxs=[tf.argmax(label_list[i]) for i in range(len(label_list))]
        for label in self.classes:
            g_idx=[]
            times=0
            clients_list=[]
            for j,d in enumerate(gen_data_idxs):
                if label==d:
                    g_idx.append(j)
            np.random.shuffle(g_idx)
            for client_name in self.cnames:
                index=int(client_name.split('_')[1])-1 
                if label in clients[index].classes:
                    clients_list.append(index)
                    times+=1
            gen_split=np.array_split(g_idx,times)  # مساوی تقسیم میشه 
            j=0
            for index in clients_list:
                if clients[index].generated_data:
                    clients[index].generated_data=clients[index].generated_data.concatenate(tf.data.Dataset.from_tensor_slices((
                                                                data_list[gen_split[j]],label_list[gen_split[j]])))
                    j+=1
                else:
                    clients[index].generated_data=tf.data.Dataset.from_tensor_slices((data_list[gen_split[j]],
                                                                                  label_list[gen_split[j]]))  
                    j+=1
# 6.
    def send_to_server(self,server):         
        if server.generated_data:
            server.generated_data=server.generated_data.concatenate(self.generated_data) 
        else:
            server.generated_data=self.generated_data         
# 7.        
    def refresh_edgeserver(self):                                               
        self.generated_data=None
        
# 8.
    def classes_registering(self,client):
        for i in client.classes:
            if i not in self.classes:
                self.classes.append(i)
