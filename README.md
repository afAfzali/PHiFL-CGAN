Implementation of the algorithm presented in the paper titled "PHiFL-CGAN: Personalized Hierarchical Federated Learning using Conditional Generative Adversarial Network" with Tensorflow.
--
* Here is one example to run this code (IID MNIST Scenario):

          dataset="mnist"
          flag1=1
          model="cnn1"  
          batch_size=32
          communication_round=6          
          epochs=20                         
          num_edge_aggregation=4           
          num_edges=3   
          num_clients=30 
          fraction_clients=0.5              
          lr=0.01
          val_ratio=0.1     
          image_shape=(28,28,1)

* Here is one example to run this code (non-IID MNIST Scenario):

        .

* Here is one example to run this code (non-IID FEMNIST Scenario):

        .

Dataset
--
Your need to download FEMNIST dataset. FEMNIST is naturally non-IID....?

**Notice:**
        You need to create the following folders where the program is located: `results\clients_models`
