# FedOV

This is source code for paper One-Shot Federated Learning by Open-Set Voting. An example running script is in `run.sh`.

Some repos we refer to

Adversarial attacks: directly use https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

Cutpaste: use an unofficial reproduction from https://github.com/Runinho/pytorch-cutpaste. The authors of Cutpaste do not make code public.

Also include code from https://github.com/lwneal/counterfactual-open-set which is the code by authors of Open Set Learning with Counterfactal Images, ECCV 2018. However, since our FL partition settings are more diverse and complicated, we find it very hard to tune the hyper-parameters to generate counterfactual inmages, so we do not call it in the final version. 
