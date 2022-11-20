# FedOV

This is source code for paper Towards Addressing Label Skews in One-shot Federated Learning. 

An example running script of FedOV is in `run.sh`. 

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `simple-cnn`, `vgg`, `resnet`, `mlp`. Default = `mlp`. |
| `dataset`      | Dataset to use. Options: `mnist`, `cifar10`, `fmnist`, `svhn`|
| `alg` | The training algorithm. Options: `vote`. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `5`. |
| `n_parties` | Number of parties, default = `2`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `partition`    | The partition way. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns)|
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `init_seed` | The initial seed, default = `0`. |

Some repos we refer to

General framework (for baseline algorithms like FedAvg, FedProx, SCAFFOLD and FedNova, please use this framework): https://github.com/Xtra-Computing/NIID-Bench

Adversarial attacks: directly use https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

Cutpaste: use an unofficial reproduction from https://github.com/Runinho/pytorch-cutpaste. The authors of Cutpaste do not make code public.

Also include code from https://github.com/lwneal/counterfactual-open-set which is the code by authors of Open Set Learning with Counterfactal Images, ECCV 2018. However, since our FL partition settings are more diverse and complicated, we find it very hard to tune the hyper-parameters to generate good counterfactual images and the voting accuracy is low, so we do not call it in the final version. 

In our code, we keep the commented or unused codes (functions). We tried these but did not get good results. After many trails and errors, we summarize the current DD and AOE functions. These trials may save efforts or bring some insights for future researchers, so we keep them. 

For ResNet-50 experiments, since ResNet-50 has batch normalization layers, we have to mix train data and generated outliers in a batch, otherwise the model will become very bad. Codes are like the following
```
x_con = torch.cat([x,x_gen11],dim=0)
y_con = torch.cat([target,y_gen],dim=0)
loss = criterion(out, y_con)
```

If you find our work useful, please cite
