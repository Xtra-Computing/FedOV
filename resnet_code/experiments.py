import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import copy
from PIL import Image
from cutpaste import *

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from attack import *

CLASSIFIER_EPOCHS = 5
GENERATIVE_EPOCHS = 1
BATCH_SIZE = 64
LATENT_SIZE = 20
NUM_CLASSES = 10

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        #self.cuda()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, latent_size)
        #self.cuda()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = norm(x)
        return x


# Project to the unit sphere
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 128)
        self.fc2 = nn.Linear(128, 4*7*7)
        self.conv1 = nn.ConvTranspose2d(4, 32, stride=2, kernel_size=4, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, stride=2, kernel_size=4, padding=1)
        #self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view(-1, 4, 7, 7)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        #self.cuda()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_generative_model(encoder, generator, discriminator, dataloader):
    generative_params = [x for x in encoder.parameters()] + [x for x in generator.parameters()]
    gen_adam = torch.optim.Adam(generative_params, lr=.005)
    disc_adam = torch.optim.Adam(discriminator.parameters(), lr=.02)
    for tmp in dataloader:
        for batch_idx, (images, labels) in enumerate(tmp):
            disc_adam.zero_grad()
            fake = generator(torch.randn(len(images), LATENT_SIZE))
            disc_loss = torch.mean(F.softplus(discriminator(fake)) + F.softplus(-discriminator(images)))
            disc_loss.backward()
            gp_loss = calc_gradient_penalty(discriminator, images, fake)
            gp_loss.backward()
            disc_adam.step()

            gen_adam.zero_grad()
            mse_loss = torch.mean((generator(encoder(images)) - images) ** 2)
            mse_loss.backward()
            gen_loss = torch.mean(F.softplus(discriminator(images)))
            #logger.info('Autoencoder loss: {:.03f}, Generator loss: {:.03f}, Disc. loss: {:.03f}'.format(
            #    mse_loss, gen_loss, disc_loss))
            gen_adam.step()
    #print('Generative training finished')


def calc_gradient_penalty(discriminator, real_data, fake_data, penalty_lambda=10.0):
    from torch import autograd
    alpha = torch.rand(real_data.size()[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    #alpha = alpha.cuda()

    # Traditional WGAN-GP
    #interpolates = alpha * real_data + (1 - alpha) * fake_data
    # An alternative approach
    interpolates = torch.cat([real_data, fake_data])
    #interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    ones = torch.ones(disc_interpolates.size())#.cuda()
    gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_lambda
    return penalty


def generate_counterfactuals(encoder, generator, classifier, dataloader):
    cf_open_set_images = []
    for tmp in dataloader:
        for batch_idx, (images, labels) in enumerate(tmp):
            counterfactuals = generate_cf( encoder, generator, classifier, images)
            cf_open_set_images.append(counterfactuals)
            if batch_idx == 0:
                gene = counterfactuals.numpy()
                np.save('0.npy', gene)
    print("Generated {} batches of counterfactual images".format(len(cf_open_set_images)))
    #imutil.show(counterfactuals, filename='example_counterfactuals.jpg', img_padding=8)
    return cf_open_set_images


def generate_cf(encoder, generator, classifier, images,
                cf_iters=1, cf_step_size=1e-2, cf_distance_weight=1.0):
    from torch.autograd import grad

    # First encode the image into latent space (z)
    z_0 = encoder(images)
    z = z_0.clone()

    # Now perform gradient descent to update z
    for i in range(cf_iters):
        # Classify with one extra class
        logits = classifier(generator(z))
        augmented_logits = F.pad(logits, pad=(0,1))

        # Use the extra class as a counterfactual target
        batch_size, num_classes = logits.shape
        target_tensor = torch.LongTensor(batch_size)#.cuda()
        target_tensor[:] = num_classes

        # Maximize classification probability of the counterfactual target
        cf_loss = F.nll_loss(F.log_softmax(augmented_logits, dim=1), target_tensor)

        # Regularize with distance to original z
        distance_loss = torch.mean((z - z_0) ** 2)

        # Move z toward the "open set" class
        loss = cf_loss + distance_loss
        dc_dz = grad(loss, z, loss)[0]
        z -= cf_step_size * dc_dz

        # Sanity check: Clip gradients to avoid nan in ill-conditioned inputs
        #dc_dz = torch.clamp(dc_dz, -.1, .1)

        # Optional: Normalize to the unit sphere (match encoder's settings)
        z = norm(z)

    #print("Generated batch of counterfactual images with cf_loss {:.03f}".format(cf_loss))
    # Output the generated image as an example "unknown" image
    return generator(z).detach()

def train_classifier(classifier, dataloader):
    adam = torch.optim.Adam(classifier.parameters())
    for tmp in dataloader:
        for batch_idx, (images, labels) in enumerate(tmp):
            adam.zero_grad()
            preds = F.log_softmax(classifier(images), dim=1)
            classifier_loss = F.nll_loss(preds, labels)
            classifier_loss.backward()
            adam.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=11)
            elif args.dataset in ("cifar100"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=101)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=11)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            if args.dataset == "cifar100":
                net = ResNet50_cifar10(num_classes=101)
            elif args.dataset == "tinyimagenet":
                net = ResNet50_cifar10(num_classes=201)
            else:
                net = ResNet50_cifar10(num_classes=11)
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

op = transforms.RandomChoice( [
    #transforms.RandomResizedCrop(sz),
    transforms.RandomRotation(degrees=(15,75)),
    transforms.RandomRotation(degrees=(-75,-15)),
    transforms.RandomRotation(degrees=(85,90)),
    transforms.RandomRotation(degrees=(-90,-85)),
    transforms.RandomRotation(degrees=(175,180)),
    #transforms.RandomAffine(0,translate=(0.2,0.2)),
    #transforms.RandomPerspective(distortion_scale=1,p=1),
    #transforms.RandomHorizontalFlip(p=1),
    #transforms.RandomVerticalFlip(p=1)
])

def cut(x):
    x_gen = copy.deepcopy(x.cpu().numpy())
    half = int(x_gen.shape[2] / 2)
    rnd = random.randint(0,5)
    pl = random.randint(0,half-1)
    pl2 = random.randint(0,half-1)
    while (abs(pl-pl2)<half/2):
        pl2 = random.randint(0,half-1)
    if rnd <= 1:
        x_gen[:,:,pl:pl+half] = x_gen[:,:,pl2:pl2+half]
    elif rnd == 2:
        x_gen[:,:,half:] = x_gen[:,:,:half]
        x_gen[:,:,:half] = copy.deepcopy(x.cpu().numpy())[:,:,half:]
    elif rnd <= 4:
        x_gen[:,pl:pl+half,:] = x_gen[:,pl2:pl2+half,:]
    else:
        x_gen[:,half:,:] = x_gen[:,:half,:]
        x_gen[:,:half,:] = copy.deepcopy(x.cpu().numpy())[:,half:,:]
    x_gen = torch.Tensor(x_gen)

    return x_gen

def rot(x):
    #rnd = random.randint(0,20)
    #if rnd < 21:
    x_gen = copy.deepcopy(x.cpu().numpy())
    half = int(x_gen.shape[2] / 2)
    pl = random.randint(0,half-1)
    rnd = random.randint(1,3)

    x_gen[:,pl:pl+half,half:] = np.rot90(x_gen[:,pl:pl+half,half:],k=rnd,axes=(1,2))
    x_gen[:,pl:pl+half,:half] = np.rot90(x_gen[:,pl:pl+half,:half],k=rnd,axes=(1,2))
    x_gen = torch.Tensor(x_gen)
    #else:
    #    x_gen = op(copy.deepcopy(x))
    #    if rnd < 20:
    #        x_gen = torch.max(x_gen, x)
    #    else:
    #        x_gen = torch.min(x_gen, x)

    return x_gen

def paint(x):
    x_gen = copy.deepcopy(x.cpu().numpy())
    size = int(x_gen.shape[2])
    sq = 4
    pl = random.randint(sq,size-sq*2)
    pl2 = random.randint(sq,size-sq-1)
    rnd = random.randint(0,1)
    if rnd == 0:
        for i in range(sq,size-sq):
            x_gen[:,i,pl:pl+sq] = x_gen[:,pl2,pl:pl+sq]
    elif rnd == 1:
        for i in range(sq,size-sq):
            x_gen[:,pl:pl+sq,i] = x_gen[:,pl:pl+sq,pl2]
    x_gen = torch.Tensor(x_gen)

    return x_gen

def blur(x):
    rnd = random.randint(0,1)
    sz = random.randint(1,4)*2+1
    sz2 = random.randint(0,2)*2+1
    if rnd == 0:
        func = transforms.GaussianBlur(kernel_size=(sz, sz2), sigma=(10, 100))
    else:
        func = transforms.GaussianBlur(kernel_size=(sz2, sz), sigma=(10, 100))
    
    return func(x)

def shuffle(x):
    rnd = random.randint(0,1)
    x_gen = copy.deepcopy(x.cpu().numpy())
    sz = x_gen.shape[0]
    li = np.split(x_gen, range(1,sz,10), axis=rnd)
    np.random.shuffle(li)
    t = np.concatenate(li, axis=rnd)
    t = torch.Tensor(t)
    return t

def train_net_vote(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, sz, num_class=10, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    '''
    classifier = Classifier(num_classes=NUM_CLASSES).to(device)
    for i in range(CLASSIFIER_EPOCHS):
        train_classifier(classifier, train_dataloader)

    encoder = Encoder(latent_size=LATENT_SIZE).to(device)
    generator = Generator(latent_size=LATENT_SIZE).to(device)
    discriminator = Discriminator().to(device)
    for i in range(GENERATIVE_EPOCHS):
        train_generative_model(encoder, generator, discriminator, train_dataloader)
    open_set_images = generate_counterfactuals(encoder, generator, classifier, train_dataloader)
    '''

    #if args_optimizer == 'adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    #elif args_optimizer == 'amsgrad':
    #    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
    #                           amsgrad=True)
    #elif args_optimizer == 'sgd':
    #    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    #device = 'cuda:7'
    criterion = nn.CrossEntropyLoss(ignore_index=11).to(device)
    #c2 = nn.CrossEntropyLoss(reduction='none').to(device)
    net.to(device)

    cnt = 0
    rnd = []

    toImg = transforms.ToPILImage()
    toTensor = transforms.ToTensor()

    op = transforms.RandomChoice( [
        #transforms.RandomResizedCrop(sz),
        transforms.RandomRotation(degrees=(15,75)),
        transforms.RandomRotation(degrees=(-75,-15)),
        transforms.RandomRotation(degrees=(85,90)),
        transforms.RandomRotation(degrees=(-90,-85)),
        transforms.RandomRotation(degrees=(175,180)),
        #transforms.RandomAffine(0,translate=(0.2,0.2)),
        #transforms.RandomPerspective(distortion_scale=1,p=1),
        #transforms.RandomHorizontalFlip(p=1),
        #transforms.RandomVerticalFlip(p=1)
    ])

    aug = transforms.Compose([
        toImg,
        op,
        toTensor
    ])

    aug_crop =  transforms.RandomChoice( [
        transforms.RandomResizedCrop(sz, scale=(0.1, 0.33)), # good
        transforms.Lambda(lambda img: blur(img)), # good
        #transforms.Lambda(lambda img: shuffle(img)), # bad
        transforms.RandomErasing(p=1, scale=(0.33, 0.5)), # good
        transforms.Lambda(lambda img: cut(img)), # fine
        transforms.Lambda(lambda img: rot(img)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
        #transforms.Lambda(lambda img: paint(img))
    ])

    attack = FastGradientSignUntargeted(net, 
                                        epsilon=0.5, 
                                        alpha=0.002, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=5,
                                        device=device)

    cp = CutPasteUnion()

    aug_final = transforms.RandomChoice( [
        transforms.Lambda(lambda img: aug_crop(img)),
        #transforms.Lambda(lambda img: cp(img)) # delete this comment if you want to add cutpaste augmentation
    ])

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):

                x, target = x.to(device), target.to(device)
                #x_gen = copy.deepcopy(x.cpu().numpy()) #open_set_images[batch_idx].to(device)
                #half = int(x_gen.shape[2] / 2)
                #x_gen[:,:,:,:half] = x_gen[:,:,:,half:]
                #x_gen = torch.Tensor(x_gen).to(device)
                y_gen = np.ones(x.shape[0]) * num_class
                y_gen = torch.LongTensor(y_gen).to(device)
                '''
                x_gen2 = copy.deepcopy(x.cpu().numpy())
                x_gen2[:,:,:,half:] = x_gen2[:,:,:,:half]
                x_gen2 = torch.Tensor(x_gen2).to(device)

                x_gen3 = copy.deepcopy(x.cpu().numpy())
                x_gen3[:,:,:,half:] = x_gen3[:,:,:,:half]
                x_gen3[:,:,:,:half] = copy.deepcopy(x.cpu().numpy())[:,:,:,half:]
                x_gen3 = torch.Tensor(x_gen3).to(device)

                x_gen4 = copy.deepcopy(x.cpu().numpy())
                x_gen4[:,:,half:,:] = x_gen4[:,:,:half,:]
                x_gen4 = torch.Tensor(x_gen4).to(device)

                x_gen5 = copy.deepcopy(x.cpu().numpy())
                x_gen5[:,:,:half,:] = x_gen5[:,:,half:,:]
                x_gen5 = torch.Tensor(x_gen5).to(device)

                x_gen6 = copy.deepcopy(x.cpu().numpy())
                x_gen6[:,:,half:,:] = x_gen6[:,:,:half,:]
                x_gen6[:,:,:half,:] = copy.deepcopy(x.cpu().numpy())[:,:,half:,:]
                x_gen6 = torch.Tensor(x_gen6).to(device)

                x_gen7 = copy.deepcopy(x.cpu().numpy())
                x_gen7[:,:,half:,half:] = np.rot90(copy.deepcopy(x.cpu().numpy())[:,:,half:,half:],axes=(2,3))
                x_gen7[:,:,half:,:half] = np.rot90(copy.deepcopy(x.cpu().numpy())[:,:,half:,:half],axes=(2,3))
                x_gen7 = torch.Tensor(x_gen7).to(device)

                x_gen8 = copy.deepcopy(x.cpu().numpy())
                x_gen8[:,:,:half,half:] = np.rot90(copy.deepcopy(x.cpu().numpy())[:,:,:half,half:],axes=(2,3))
                x_gen8[:,:,:half,:half] = np.rot90(copy.deepcopy(x.cpu().numpy())[:,:,:half,:half],axes=(2,3))
                x_gen8 = torch.Tensor(x_gen8).to(device)
                '''
                '''
                x_gen9 = copy.deepcopy(x.cpu().numpy())
                for i in range(x_gen9.shape[0]):
                    x_gen9[i] = aug(torch.Tensor(x_gen9[i]))
                x_gen9 = torch.Tensor(x_gen9).to(device)
                x_gen9 = torch.max(x_gen9, x)
                
                x_gen10 = copy.deepcopy(x.cpu().numpy())
                for i in range(x_gen10.shape[0]):
                    x_gen10[i] = aug(torch.Tensor(x_gen10[i]))
                x_gen10 = torch.Tensor(x_gen10).to(device)
                x_gen10 = torch.min(x_gen10, x)
                '''
                x_gen11 = copy.deepcopy(x.cpu().numpy())
                for i in range(x_gen11.shape[0]):
                    x_gen11[i] = aug_final(torch.Tensor(x_gen11[i]))
                x_gen11 = torch.Tensor(x_gen11).to(device)
                
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                #x_gen.requires_grad = True
                y_gen.requires_grad = False
                #x_gen2.requires_grad = True
                #x_gen3.requires_grad = True
                #x_gen4.requires_grad = True
                #x_gen5.requires_grad = True
                #x_gen6.requires_grad = True
                #x_gen7.requires_grad = True
                #x_gen8.requires_grad = True
                #x_gen9.requires_grad = True
                #x_gen10.requires_grad = True
                x_gen11.requires_grad = True
                #x_gen12.requires_grad = True

                # out, mid = net(x) 
                #out_gen11, _ = net(x_gen11)
                '''
                out_gen = net(x_gen)
                out_gen2 = net(x_gen2)
                out_gen3 = net(x_gen3)
                out_gen4 = net(x_gen4)
                out_gen5 = net(x_gen5)
                out_gen6 = net(x_gen6)
                out_gen7 = net(x_gen7)
                out_gen8 = net(x_gen8)
                '''
                #out_gen9 = net(x_gen9)
                #out_gen10 = net(x_gen10)
                # out_gen11, _ = net(x_gen11)


                x_con = torch.cat([x,x_gen11],dim=0)
                y_con = torch.cat([target,y_gen],dim=0)
                out, _ = net(x_con)
                loss = criterion(out, y_con)


                #out_gen12 = net(x_gen12)
                
                # one_hot = torch.zeros(out.cpu().shape[0], out.cpu().shape[1]).scatter_(1, target.cpu().reshape(-1, 1), 1)
                # one_hot = one_hot.to(device)
                # out_second = out - one_hot * 10000

                # ind = np.arange(x.shape[0])
                # np.random.shuffle(ind)
                # y_mask = np.arange(x.shape[0])
                # labels = target.cpu().numpy()
                # y_mask = np.where(labels[y_mask] == labels[ind[y_mask]], 11, 10)
                
                '''
                if epoch == 0:
                    co = random.gauss(0.5, 0.5)
                    if co < 0:
                        co = 0
                    rnd.append(co)
                else:
                    co = rnd[batch_idx]
                '''
                # loss = criterion(out, target) + criterion(out_gen11, y_gen) + criterion(out_second, y_gen) #+ criterion(out_gen9, y_gen) + criterion(out_gen10, y_gen) #criterion(out_gen, y_gen) + criterion(out_gen2, y_gen) + criterion(out_gen3, y_gen) + criterion(out_gen4, y_gen) + criterion(out_gen5, y_gen) + criterion(out_gen6, y_gen) + criterion(out_gen7, y_gen) + criterion(out_gen8, y_gen) #+ criterion(out_gen12, y_gen)
                
                #if np.min(y_mask) == 10:                    
                #    y_mask = torch.LongTensor(y_mask).to(device)

                    # beta=torch.distributions.beta.Beta(1, 1).sample([]).item()
                    # mixed_embeddings = beta * mid + (1-beta) * mid[ind]
                    # mixed_out = net.later_layers(mixed_embeddings) 
                    # loss += criterion(mixed_out, y_mask) * 0.01
                
                #adv_data = attack.perturb(x_gen11, y_gen)

                #out_adv, _ = net(adv_data)

                #loss += criterion(out_adv, y_gen)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        #auc = compute_auc_outlier_detection(net_id, net, test_dataloader, device=device) #can be used to perform traditional outlier detection experiments, calculate ROC-AUC under noniid-#label1 partition
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    #device = 'cpu'
    #net.to(device)
    '''
    flag = False
    for tmp in train_dataloader:
        for batch_idx, (x, target) in enumerate(tmp):
            x_gen11 = copy.deepcopy(x.cpu().numpy())
            for i in range(x_gen11.shape[0]):
                x_gen11[i] = aug_crop(torch.Tensor(x_gen11[i]))
            x_gen11 = torch.Tensor(x_gen11).to(device)

            out, mid = net(x_gen11)

            if not flag:
                flag = True
                outliers = mid.cpu().detach().numpy()
            else:
                outliers = np.concatenate((outliers,mid.cpu().detach().numpy()))
    '''
    train_acc, threshold, max_prob, avg_max = compute_accuracy(net, train_dataloader, calc=True, device=device)
    test_acc = compute_accuracy(net, test_dataloader, device=device)#, add=outliers)
    
    logger.info(threshold)
    logger.info(max_prob)
    logger.info(avg_max)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    logger.info(' ** Training complete **')
    return threshold, max_prob, avg_max


def local_train_net_vote(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    threshold_list = []

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if args.dataset in ('mnist', 'fmnist'):
            sz = 28
        else:
            sz = 32
        
        num_class = 10
        if args.dataset == 'cifar100':
            num_class = 100
        elif args.dataset == 'tinyimagenet':
            num_class = 200
        
        threshold, max_prob, avg_max = train_net_vote(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, sz, num_class=num_class, device=device)
        threshold_list.append([float(threshold), float(max_prob), float(avg_max)])
       
    return threshold_list


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)


    if args.alg == 'vote':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        threshold_list=[]
        threshold_list = local_train_net_vote(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)
        #logger.info(threshold_list)

        model_list = [net for net_id, net in nets.items()]
        
        #train_acc = compute_accuracy_vote(nets, train_dl_global)
        for factor in [1]:
            logger.info("Factor = {}".format(factor))
            #logger.info("Normalize")
            #for accepted_vote in range(1, 11):
            #    test_acc = compute_accuracy_vote(model_list, threshold_list, test_dl_global, accepted_vote, factor=factor,device=device)
            #    logger.info("Max {} vote: test acc = {}".format(accepted_vote, test_acc))
            
            logger.info("Not Normalize")
            for accepted_vote in range(1, 11):
                test_acc, half, pred_labels_list = compute_accuracy_vote_soft(model_list, threshold_list, test_dl_global, accepted_vote, normalize = False, factor=factor,device=device)
                logger.info("Max {} vote: test acc = {}".format(accepted_vote, test_acc))
            #logger.info(half)
            #logger.info(pred_labels_list.shape)
            #logger.info(pred_labels_list)

        stu_nets = init_nets(args.net_config, args.dropout_p, 1, args)
        stu_model = stu_nets[0][0]
        distill_soft(stu_model, pred_labels_list, test_dl_global, half, args=args, device=device)
        # compute_accuracy_vote_soft() and distill_soft() for soft label distillation like FedDF. 
        # compute_accuracy_vote() and distill() are hard label distillation.
        # Soft label is usually better, especially for complicated datasets like CIFAR-10, CIFAR-100.
            
