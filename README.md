# FedOV

This is source code for paper One-Shot Federated Learning by Open-Set Voting. An example running script is in `run.sh`.

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

General framework: https://github.com/Xtra-Computing/NIID-Bench

Adversarial attacks: directly use https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

Cutpaste: use an unofficial reproduction from https://github.com/Runinho/pytorch-cutpaste. The authors of Cutpaste do not make code public.

Also include code from https://github.com/lwneal/counterfactual-open-set which is the code by authors of Open Set Learning with Counterfactal Images, ECCV 2018. However, since our FL partition settings are more diverse and complicated, we find it very hard to tune the hyper-parameters to generate counterfactual inmages, so we do not call it in the final version. 
