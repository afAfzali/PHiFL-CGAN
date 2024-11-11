Implementation of the algorithm presented in the paper titled "Private and Heterogeneous Personalized Hierarchical Federated Learning using Conditional Generative Adversarial Networks" with Tensorflow.
--
* Here is one example to run this code (IID MNIST Scenario):

          dataset="mnist"
          flag1=1
          model="cnn1"  
          batch_size=32
          communication_round=3          
          epochs=10                         
          num_edge_aggregation=4
          cgan_round=30
          num_edges=3   
          num_clients=30 
          lr=0.01      # for classifier model 
          image_shape=(28,28,1)
          train_size=21000
          test_size=9000
          latent_dim=100
          gen_ratio=0.5
          lr=0.0002    # for CGAN
          beta_1=0.5   # for CGAN

* Here is one example to run this code (non-IID MNIST Scenario):

          dataset="mnist"
          flag1=5
          model="cnn1"  
          batch_size=32
          communication_round=3          
          epochs=10                         
          num_edge_aggregation=4
          cgan_round=30
          num_edges=3   
          num_clients=30 
          lr=0.01      # for classifier model 
          image_shape=(28,28,1)
          train_size=21000
          test_size=9000
          latent_dim=100
          gen_ratio=0.5
          lr=0.0002    # for CGAN
          beta_1=0.5   # for CGAN

* Here is one example to run this code (non-IID FEMNIST Scenario):

          dataset="femnist"
          model="cnn1"  
          batch_size=32
          communication_round=3          
          epochs=10                         
          num_edge_aggregation=4
          cgan_round=30
          num_edges=3   
          num_clients=30 
          lr=0.01      # for classifier model 
          image_shape=(28,28,1)
          train_size=21000
          test_size=9000
          latent_dim=100
          gen_ratio=0.5
          lr=0.0005    # for CGAN
          beta_1=0.5   # for CGAN

Dataset
--
Your need to download FEMNIST dataset. FEMNIST is naturally non-IID....?

**Notice:**
        You need to create the following folders where the program is located: `results\clients_models`
