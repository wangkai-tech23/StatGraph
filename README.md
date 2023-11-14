# StatGraph
StatGraph is multi-view statistical graph learning model for effective and fined-grained in-vehicle network intrusion detection. With high-performance operation efficiency, StatGraph is well adapted to the vehicle environment and protects the security of the in-vehicle CAN bus. 

You can see the details in our paper **Effective In-vehicle Intrusion Detection via Multi-view Statistical Graph Learning on CAN Messages**. ([arXiv:2311.07056](https://arxiv.org/abs/2311.07056))

We provide three folders: the "Dataset" folder for the raw data, the "StatGraph-CarHacking" folder and the "StatGraph-ROAD" folder for the data processing code and the neural network related code. 

Please execute the code in each "dataprocess" folder inside each "StatGraph-" folder to generate data that can be fed into the neural network, before running the code in "ModelAdapting/run" folder.
