import numpy as np
import pickle
import tracemalloc
import random
import os
import cupy as cp
import psutil
import shutil
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numexpr as ne
import time
import matplotlib.pyplot as plt
import gc
import sys
import ctypes
import tensorflow as tf
from client import Client
from edgeserver import Edgeserver
from server import Server 
from datasets_partitioning.mnist_femnist import get_dataset
from datasets_partitioning.mnist_femnist import k_niid_equal_size_split
from datasets_partitioning.mnist_femnist import k_niid_equal_size_split_1
from datasets_partitioning.mnist_femnist import Gaussian_noise
from datasets_partitioning.mnist_femnist import get_classes
from datasets_partitioning.mnist_femnist import random_edges
from datasets_partitioning.mnist_femnist import iid_edges
from datasets_partitioning.mnist_femnist import niid_edges
from datasets_partitioning.mnist_femnist import iid_equal_size_split
from datasets_partitioning.mnist_femnist import iid_nequal_size_split
from datasets_partitioning.mnist_femnist import niid_labeldis_split
from datasets_partitioning.mnist_femnist import equal_size_split
from datasets_partitioning.mnist_femnist import get_clients_femnist_cnn_with_reduce_writers_k_classes
from tensorflow.keras.models import load_model
from model.initialize_model import create
from tensorflow.keras.utils import plot_model,to_categorical

# =============================================================================================================
#                                                Partitioning                
# =============================================================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)  
tf.keras.backend.clear_session()
dataset="femnist"        #"femnist" or "mnist"
if dataset=='cifar10' or dataset=="mnist":
    number_labels=10
if dataset=='femnist':
    number_labels=10    # number classes of 62 classes 
model="cnn1"   #or cnn1 , cnn2, cnn3
batch_size=32
communication_round=3 
epochs=10                       #  number of local update 
num_edge_aggregation=4          #  number of edge aggregation 
num_edges=3   
num_clients=30 
#fraction_clients=1            
lr=0.01
#val_ratio=0.1     
beta=0.5        
gen_ratio=2          
cgan_round=30
train_size=21000
test_size=9000
latent_dim=100
image_shape=(28,28,1)
loss="categorical_crossentropy"  #optimizer is "Adam"
metrics=["accuracy"]
verbose=2    
seed=5  
np.random.seed(seed)
random.seed(seed)
optimizer=tf.keras.optimizers.SGD(learning_rate=lr)

#     ********** Get dataset **********
tracemalloc.start()
process=psutil.Process()
start_rss=process.memory_info().rss

#     ********** partitioning and assigning ********** 
if dataset!="femnist":
    X_train ,Y_train,X_test,Y_test=get_dataset(dataset,model) 
    X_train ,Y_train,X_test,Y_test=X_train[:train_size] ,Y_train[:train_size],X_test[:test_size],Y_test[:test_size]
    print('1 : clients_iid (equal size)\n'
          '2 : clients_iid (nonequal size)\n'
          '3 : each client owns data samples of a fixed number of labels\n'
          '4 : each client(and edge) owns data samples of a different feature distribution\n'
          '5 : each client owns a proportion of the samples of each label\n')
    flag1=int(input('select a number:')) 
    print("\nUsing a locally saved model?\n"
            "1 : YES\n"
            "0 : NO\n")
    replace=int(input('select a number:'))
    #     ***********clients_iid*****************
    if flag1 in (1,2):                                    
        print('\n** randomly are assigned clients to edgesevers **')
        clients=[]
        edges=[]
        if flag1==1:
            train_partitions=iid_equal_size_split(X_train,Y_train,num_clients)
            test_partitions=iid_equal_size_split(X_test,Y_test,num_clients)
        else:
            train_partitions=iid_nequal_size_split(X_train,Y_train,num_clients,beta)
            test_partitions=iid_nequal_size_split(X_test,Y_test,num_clients,beta)
        for i in range(num_clients):
            client_classes=get_classes(train_partitions[i],number_labels)
            clients.append(Client(i,train_partitions[i],test_partitions[i],client_classes,dataset,model,loss,metrics,
                                                             lr,image_shape,latent_dim,number_labels,batch_size))     
        assigned_clients_list=random_edges(num_edges,num_clients) 
        for edgeid in range(num_edges):
            edges.append(Edgeserver(edgeid,assigned_clients_list[edgeid],dataset,image_shape,latent_dim,number_labels))
            for client_name in assigned_clients_list[edgeid]:               
                index=int(client_name.split('_')[1])-1                # k-1
                edges[edgeid].classes_registering(clients[index])
        clients_per_edge=int(num_clients/num_edges)
        server=Server()   
        print(tracemalloc.get_traced_memory()) 
        del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions,assigned_clients_list
        gc.collect()
        print(tracemalloc.get_traced_memory()) 
      
    #     ********** each edge owns data samples of a fixed number of labels ********** 
    elif flag1==3:                                        
        clients_per_edge=int(num_clients/num_edges)
        k1=int(input('\nk1 : number of labels for each edge  ?  '))
        k2=int(input('k2 : number of labels for clients per edge  ?  '))
        print(f'\n** assign each edge {clients_per_edge} clients with {k1} classes'
              f'\n** assign each client samples of {k2}  classes of {k1} edge classes')
        label_list=list(range(number_labels))
        X_train,Y_train,X_test,Y_test,party_labels_list=k_niid_equal_size_split(X_train,Y_train,X_test,
                                                                            Y_test,num_edges,label_list,k1,flag1)   
        clients=[]
        edges=[]
        index=0  
        for edgeid in range(num_edges):           
            train_partitions,test_partitions=k_niid_equal_size_split(X_train[edgeid],Y_train[edgeid],X_test[edgeid],
                                                    Y_test[edgeid],clients_per_edge,party_labels_list[edgeid],k2)
            assigned_clients=[]
            for i in range(clients_per_edge):
                client_classes=get_classes(train_partitions[i],number_labels)
                clients.append(Client(index,train_partitions[i],test_partitions[i],client_classes,dataset,model,loss,
                                                               metrics,lr,image_shape,latent_dim,number_labels,batch_size))   
                assigned_clients.append(index)
                index+=1
            assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
            edges.append(Edgeserver(edgeid,assigned_clients,dataset,image_shape,latent_dim,number_labels))
            for client_name in assigned_clients:                 # client's name : 'client_k'
                idx=int(client_name.split('_')[1])-1                # k-1
                edges[edgeid].classes_registering(clients[idx])
            for i in range(clients_per_edge):
                print(f'{edges[edgeid].cnames[i]}')
            print(f'be assigned to {edges[edgeid].name}')
        server=Server()   
        print(tracemalloc.get_traced_memory()) 
        del X_train,X_test,Y_train,Y_test,test_partitions,train_partitions
        gc.collect()  
        print(tracemalloc.get_traced_memory()) 

    #     ********** each edge owns data samples of a different feature distribution ********** 
    #     ***** each edge owns data samples of 10 labels but each client owns data samples of one or 10 labels ***** 
    elif flag1==4:                                              
        original_std=float(input('\noriginal standard deviation for gaussian noise  ?  '))
        k=int(input('k : number of labels for clients of each edge  ?  '))  
        X_train,Y_train=iid_equal_size_split(X_train,Y_train,num_edges,flag1) 
        X_test,Y_test=iid_equal_size_split(X_test,Y_test,num_edges,flag1)
        edges=[]
        clients=[]
        clients_per_edge=int(num_clients/num_edges)
        labels_list=list(range(number_labels)) 
        mean=0       
        index=0 
        for edgeid in range(num_edges):
            train_noisy_edge=Gaussian_noise(X_train[edgeid],original_std,edgeid,num_edges,mean)
            test_noisy_edge=Gaussian_noise(X_test[edgeid],original_std,edgeid,num_edges,mean)
            train_party_partitions,test_party_partitions=k_niid_equal_size_split(train_noisy_edge,Y_train,test_noisy_edge, 
                                                                                 Y_test,clients_per_edge,labels_list,k)
            assigned_clients=[]
            for i in range(clients_per_edge):
                client_classes=get_classes(train_party_partitions[i],number_labels)
                clients.append(Client(index,train_party_partitions[i],test_party_partitions[i],client_classes,dataset,
                                                     model,loss,metrics,lr,image_shape,latent_dim,number_labels,batch_size))  
                assigned_clients.append(index)
                index+=1
            assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
            edges.append(Edgeserver(edgeid,assigned_clients,dataset,image_shape,latent_dim,number_labels))
            for client_name in assigned_clients:                  
                idx=int(client_name.split('_')[1])-1                
                edges[edgeid].classes_registering(clients[idx])
            for i in range(clients_per_edge):
                print(f'{edges[edgeid].cnames[i]}')
            print(f'be assigned to {edges[edgeid].name}')
        server=Server()   
        print(tracemalloc.get_traced_memory()) 
        del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions,train_noisy_edge,test_noisy_edge,train_party_partitions,test_party_partitions
        gc.collect()
        print(tracemalloc.get_traced_memory())

    #     ************** each client owns a proportion of the samples of each label **************
    elif flag1==5:                    
        train_partitions=niid_labeldis_split(X_train,Y_train,num_clients,'train',beta)
        test_partitions=niid_labeldis_split(X_test,Y_test,num_clients,'test',beta)
        clients=[]
        edges=[]
        clients_per_edge=int(num_clients/num_edges)
        index=0  
        for edgeid in range(num_edges):                           
            assigned_clients=[]
            for _ in range(clients_per_edge):
                client_classes=get_classes(train_partitions[index],number_labels)
                clients.append(Client(index,train_partitions[index],test_partitions[index],client_classes,dataset,model,loss,
                                                    metrics,lr,image_shape,latent_dim,number_labels,batch_size))  
                assigned_clients.append(index)
                index+=1
            assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
            edges.append(Edgeserver(edgeid,assigned_clients,dataset,image_shape,latent_dim,number_labels))
            for client_name in assigned_clients:                 
                idx=int(client_name.split('_')[1])-1               
                edges[edgeid].classes_registering(clients[idx])
            for i in range(clients_per_edge):
                print(f'{edges[edgeid].cnames[i]}')
            print(f'be assigned to {edges[edgeid].name}')
        server=Server()   
        print(tracemalloc.get_traced_memory()) 
        del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions
        gc.collect()
        print(tracemalloc.get_traced_memory()) 

elif dataset=="femnist": 
    print('equal size + reducing writers')
    print('\n** randomly are assigned clients to edgesevers **')
    print("\nUsing a locally saved model?\n"
            "1 : YES\n"
            "0 : NO\n")
    replace=int(input('select a number:'))
    train_partitions,test_partitions=get_clients_femnist_cnn_with_reduce_writers_k_classes(num_clients,train_size,
                                                                                           test_size,number_labels)
    print("partitinong ...end !")
    clients=[]
    edges=[]
    for i in range(num_clients):
        client_classes=get_classes(train_partitions[i],number_labels)
        clients.append(Client(i,train_partitions[i],test_partitions[i],client_classes,dataset,model,loss,metrics,
                                                     lr,image_shape,latent_dim,number_labels,batch_size))     
    assigned_clients_list=random_edges(num_edges,num_clients) 
    for edgeid in range(num_edges):
        edges.append(Edgeserver(edgeid,assigned_clients_list[edgeid],dataset,image_shape,latent_dim,number_labels))
        for client_name in assigned_clients_list[edgeid]:               
            index=int(client_name.split('_')[1])-1                # k-1
            edges[edgeid].classes_registering(clients[index])
    clients_per_edge=int(num_clients/num_edges)
    server=Server()   
    print(tracemalloc.get_traced_memory()) 
    del train_partitions,test_partitions,assigned_clients_list
    gc.collect()
    print(tracemalloc.get_traced_memory())

    # =============================================================================================================
if replace==0:
    path=r'.\results\clients_models\\'                        
    for file_name in os.listdir(path):
        file=path+file_name
        if os.path.isfile(file):
            #print(f'deleting file:{file_name}')
            os.remove(file)
start_time=time.time()
for comm_r in range(communication_round):    
    print(f'===================================={comm_r+1} c_round...start================================================')
    # generated_data is cleared
    server.refresh_server()
    # my assumption: all edges participate in training phase in each communication round             
    for num_agg in range(num_edge_aggregation+1):
        print(f'--------------------------------------{num_agg+1} agg...start---------------------------------------') 
        for edge in edges:
            print(f'************{edge.name}******************start')
            # generated_data is cleared
            edge.refresh_edgeserver()
            for client_name in edge.cnames:  
                print(f"\n--------------------------------> {client_name} be selected:")
                index=int(client_name.split('_')[1])-1
                # classifier training
                #clients[index].m_compile(loss=loss,optimizer=optimizer,metrics=metrics)   
                if comm_r!=0 or num_agg!=0:
                    clients[index].local_model_train(epochs,batch_size,verbose)    
                else:
                    if replace==1:
                        clients[index].classifier=load_model(fr".\results\clients_models\{clients[index].name}.h5")
                    else:
                        clients[index].local_model_train(epochs,batch_size,verbose)
                        clients[index].classifier.save(fr".\results\clients_models\{clients[index].name}.h5") 
                clients[index].compute_test()
                for cgan_r in range(cgan_round):
                    real_data=clients[index].get_data(batch_size)
                    for X_real,Y_real in real_data:
                    # edge' generator generates fake data and fake label for client
                        X_fake,Y_fake=edge.generate_fake_data(clients[index],batch_size,latent_dim,number_labels)
                        # discriminator training
                        clients[index].train_disc(X_real,Y_real,X_fake,Y_fake,batch_size) 
                        clients[index].send_to_edgeserver(edge)
                        # generator training
                        pinned_mempool=manage_pinned_memory()
                        edge.train_generator(clients[index],batch_size,latent_dim,number_labels)
                # generated_data' list updating           
                edge.update_generated_data(clients[index],gen_ratio,latent_dim,number_labels) 
                gc.collect()
                # generated_data is cleared
                clients[index].refresh_client()
            print(f'************{edge.name}******************end')
            if num_agg!=num_edge_aggregation:             #-1:
                edge.distribute_between_clients(clients) 
                gc.collect()
        print(f'--------------------------------------{num_agg+1} agg...end---------------------------------------') 
    for edge in edges:
        edge.send_to_server(server)
        edge.refresh_edgeserver()
    server.distribute_between_edgeservers(edges,number_labels) 
    for edge in edges:
        edge.distribute_between_clients(clients)
        gc.collect()
    print(f'===================================={comm_r+1} c_round...end================================================')        
for client in clients:
    client.local_model_train(epochs,batch_size,verbose)
    client.compute_test()
    #logging.info(f'--------memory after {comm_r+1}_comm:{tracemalloc.get_traced_memory()}-----------')
training_time=time.time()-start_time     
print('time',training_time)
#print(process.memory_percent())
print(process.memory_info().rss-start_rss)
print(tracemalloc.get_traced_memory())
tracemalloc.stop()
# -------------------------------------
for client in clients:
    print(client.name,":",client.test_acc)


