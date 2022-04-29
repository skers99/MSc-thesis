#Zenke's tools for generating a spiking dataset for randman
import numpy as np
import pandas as pd
import sys

if '/Users/svenkerstjens/DataSpell/lib/python3.9/site-packages' not in sys.path:
    sys.path.append('/Users/svenkerstjens/DataSpell/lib/python3.9/site-packages')
import torch
import snn_utils
from randman.randman import *
import matplotlib.pyplot as plt
import seaborn as sns


def standardize(x,eps=1e-7):
    mi,_ = x.min(0)
    ma,_ = x.max(0)
    return (x-mi)/(ma-mi+eps)

def make_spiking_dataset(nb_classes=10, nb_units=100, nb_steps=100, step_frac=1.0, dim_manifold=2, nb_spikes=1, nb_samples=1000, alpha=2.0, shuffle=True, classification=True, seed=None):
    """ Generates event-based generalized spiking randman classification/regression dataset.
    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding won't work.
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.
    Args:
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
        seed: The random seed (default: None)
    Returns:
        A tuple of data,labels. The data is structured as numpy array
        (sample x event x 2 ) where the last dimension contains
        the relative [0,1] (time,unit) coordinates and labels.
    """

    data = []
    labels = []
    targets = []

    if seed is not None:
        np.random.seed(seed)

    max_value = np.iinfo(int).max
    randman_seeds = np.random.randint(max_value, size=(nb_classes,nb_spikes) )

    for k in range(nb_classes):
        x = np.random.rand(nb_samples,dim_manifold)
        submans = [ Randman(nb_units, dim_manifold, alpha=alpha, seed=randman_seeds[k,i]) for i in range(nb_spikes) ]
        units = []
        times = []
        for i,rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(np.repeat(np.arange(nb_units).reshape(1,-1),nb_samples,axis=0))
            times.append(y.numpy())

        units = np.concatenate(units,axis=1)
        times = np.concatenate(times,axis=1)
        events = np.stack([times,units],axis=2)
        data.append(events)
        labels.append(k*np.ones(len(units)))
        targets.append(x)

    data = np.concatenate(data, axis=0)
    labels = np.array(np.concatenate(labels, axis=0), dtype=int)
    targets = np.concatenate(targets, axis=0)

    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]
        targets = targets[idx]

    data[:,:,0] *= nb_steps*step_frac
    # data = np.array(data, dtype=int)

    if classification:
        return data, labels
    else:
        return data, targets

def show_samples(data,labels):
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1,5)
    fig = plt.figure(figsize=(7,1.8),dpi=100)

    for i in range(5):
        ax = fig.add_subplot(gs[i])
        ax.scatter(data[i,:,0], data[i,:,1], marker=".", color=sns.color_palette()[labels[i]]
                   )
        ax.set_xlabel("Time")
        if i==0: ax.set_ylabel("Neuron")

    plt.tight_layout()
    sns.despine()

def prep_labels(labels):

    return  torch.Tensor(labels)

def prep_data(data,T):
    '''
    takes randman data and makes it one hot
    :param data:
    :return:
    '''
    N=data.shape[1]
    data_new = np.empty((len(data),T,N))
    for i in range(len(data)):
        temp = np.zeros((T,N))
        temp[data[i][:,0],data[i][:,1]]=1
        data_new[i]=temp
    return torch.Tensor(data_new)

def instance_to_one_hot(instance, T=1000,N=100):
    '''
    Takes an instance from randman dataset and returns one hot encoding
    :param instance: one training instance from randman spiking dataset ( as IntTensor)
    :param T: assumed time per spike train
    :param N: assumed number of neurons
    :return: FloatTensor of one hot represenations
    '''

    b = np.zeros((T,N))
    b[instance[:,0],instance[:,1]]=1
    return torch.FloatTensor(b)

def data_to_one_hot(data,T=1000,N=100):
    '''
    applies instance_to_one_hot to dataset
    :param data:
    :return:
    '''
    all = torch.Tensor()
    for i in range(len(data)):
        one_hot_instance = instance_to_one_hot(data[i],T,N).unsqueeze(0)
        all = torch.cat((all,one_hot_instance),0)
        if i % 100 ==0:
            print(f'{i}...')
    return all

def labels_to_one_hot(labels):
    '''
    takes labels and produces one-hot encoding,
    necessary for 'labels_to_spiketrains()'
    :param labels:
    :return:
    '''
    b = np.zeros((len(labels), labels.max()+1))
    b[np.arange(len(labels)),labels]=1
    return torch.FloatTensor(b)

def labels_to_spiketrains(labels,T=1000,convolve=True):
    '''

    :param labels: labels from Randman spiking dataset
    :param T: timesteps per spike train
    :param convolve: if True convolve spikes
    :return: target spike trains with spike-time encoding where each output neuron encodes one class
    '''

    spiketrains = torch.FloatTensor()
    hot_labels = labels_to_one_hot(labels)
    t_hot_labels = np.array((hot_labels-1)*-(T-1),dtype='int')

    for i in range(len(t_hot_labels)):
        one_hot_label = np.zeros((t_hot_labels[i].max()+1,len(t_hot_labels[i])))
        one_hot_label[t_hot_labels[i],np.arange(len(t_hot_labels[i]))] = 1
        one_hot_label = torch.FloatTensor(one_hot_label).unsqueeze(0)
        spiketrains = torch.cat((spiketrains,one_hot_label),0)
        if i%100==0:
            print(f'{i}...')

    if convolve:
        temp = np.zeros_like(spiketrains)
        for i in range(spiketrains.shape[0]): # for each instance
            for j in range(spiketrains.shape[-1]): # for each neuron
                temp[i,:,j] = np.convolve(spiketrains[i,:,j],np.exp(-np.linspace(0,1,100)/.1))[:spiketrains.shape[1]] # over time steps
        spiketrains = temp
    return spiketrains

def instance_to_one_hot(instance, T=1000,N=100):
    '''
    Takes an instance from randman dataset and returns one hot encoding
    :param instance: one training instance from randman spiking dataset ( as IntTensor)
    :param T: assumed time per spike train
    :param N: assumed number of neurons
    :return: FloatTensor of one hot represenations
    '''

    b = np.zeros((T,N))
    b[instance[:,0],instance[:,1]]=1
    return torch.FloatTensor(b)

def data_to_one_hot(data,T=1000,N=100):
    '''
    applies instance_to_one_hot to dataset
    :param data:
    :return:
    '''
    all = torch.Tensor()
    for i in range(len(data)):
        one_hot_instance = instance_to_one_hot(data[i],T,N).unsqueeze(0)
        all = torch.cat((all,one_hot_instance),0)

    return all

def labels_to_one_hot(labels):
    '''
    takes labels and produces one-hot encoding,
    necessary for 'labels_to_spiketrains()'
    :param labels:
    :return:
    '''
    b = np.zeros((len(labels), labels.max()+1))
    b[np.arange(len(labels)),labels]=1
    return torch.FloatTensor(b)

def labels_to_spiketrains(labels,T=1000,convolve=True):
    '''

    :param labels: labels from Randman spiking dataset
    :param T: timesteps per spike train
    :param convolve: if True convolve spikes
    :return: target spike trains with spike-time encoding where each output neuron encodes one class
    '''

    spiketrains = torch.FloatTensor()
    hot_labels = labels_to_one_hot(labels)
    t_hot_labels = np.array((hot_labels-1)*-(T-1),dtype='int')

    for i in range(len(t_hot_labels)):
        one_hot_label = np.zeros((t_hot_labels[i].max()+1,len(t_hot_labels[i])))
        one_hot_label[t_hot_labels[i],np.arange(len(t_hot_labels[i]))] = 1
        one_hot_label = torch.FloatTensor(one_hot_label).unsqueeze(0)
        spiketrains = torch.cat((spiketrains,one_hot_label),0)
        if i%100==0:
            print(f'{i}...')

    if convolve:
        temp = np.zeros_like(spiketrains)
        for i in range(spiketrains.shape[0]): # for each instance
            for j in range(spiketrains.shape[-1]): # for each neuron
                temp[i,:,j] = np.convolve(spiketrains[i,:,j],np.exp(-np.linspace(0,1,100)/.1))[:spiketrains.shape[1]] # over time steps
        spiketrains = temp
    return spiketrains

def get_randman_data(nb_classes=2, nb_units=1000, nb_steps=100, step_frac=1.0, dim_manifold=2, nb_spikes=1, nb_samples=1000, alpha=2.0, shuffle=True, classification=True, seed=None,):
    data,labels = make_spiking_dataset(nb_classes=nb_classes,nb_units=nb_units,
                                       nb_steps=nb_steps,step_frac=step_frac,
                                       dim_manifold=dim_manifold,nb_spikes=nb_spikes,
                                       nb_samples=nb_samples,alpha=alpha,shuffle=shuffle,
                                       classification=classification,seed=seed)
    data = torch.IntTensor(data)
    data_one_hot = data_to_one_hot(data,T=nb_steps,N=nb_units)
    labels_one_hot = labels_to_one_hot(labels)

    return data_one_hot,labels_one_hot

def get_batches(data,batch_size=10):


    data_points = len(data)
    num_batches = int(data_points/batch_size)

    batch_ids = np.random.choice(data_points,(num_batches,batch_size),replace=False)
    print(f'created {len(batch_ids)} batches')

    return batch_ids


