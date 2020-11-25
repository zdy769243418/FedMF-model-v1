## FedMF

#### Parameter setting

The parameters are declared in ```shared_parameters.py```, which are used by other files.



#### Distributed matrix factorization

Distributed MF are implemented in ```DistributedMF_Full.py``` and ```DistributedMF_Part.py``` 
which are full and part versions.

#### Federated matrix factorization

FedMF are implemented in ```FedMF_Full.py``` and ```FedMF_Part.py``` 
which are full and part versions.

#### step1 读懂代码，将其改成三层架构，这一步先不考虑辅助信息，然后做实验验证时间效率的提升和模型准确度的变化
#### step2 在三层架构的基础上为不同用户分配不同的模型权重，然后做实验验证时间效率的提升和模型准确度的变化
#### step2 在三层架构的基础上，构建知识图谱，把修改模型，添加辅助信息，然后做实验验证时间效率的提升和模型准确度的变化