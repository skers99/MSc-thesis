import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple

class LIFDenseNeuronState(nn.Module):
    """
    Generic module for storing the state of an RNN/SNN.
    We use the buffer function of torch nn.Module to register our
    different states such that PyTorch can manage them.
    """
    def __init__(self, in_channels, out_channels):
        """Simple initialization of the internal states of a LIF population."""
        super(LIFDenseNeuronState, self).__init__()
        self.state_names = ['U', 'I', 'S']
        self.register_buffer('U',torch.zeros(1, out_channels), persistent=True)
        self.register_buffer('I',torch.zeros(1, out_channels), persistent=True)
        self.register_buffer('S',torch.zeros(1, out_channels), persistent=True)
                                                    
    def update(self, **values):
        """Function to update the internal states."""
        for k, v in values.items():
            setattr(self, k, v) 
    
    def init(self, v=0): 
        """Function that detaches the state/graph across trials."""
        for k in self.state_names:
            state = getattr(self, k)
            setattr(self, k, torch.zeros_like(state)+v)

class LIFDensePopulation(nn.Module):
    # NeuronState = namedtuple('NeuronState', ['U', 'I', 'S'])
    def __init__(self, in_channels, out_channels, bias=True, alpha = .9, beta=.85):
        super(LIFDensePopulation, self).__init__()
        self.fc_layer = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_scale = 0.2
        self.alpha = alpha
        self.beta = beta
        #self.state = NeuronState(U=torch.zeros(batch_size, out_channels).to(self.device),
        #                         I=torch.zeros(batch_size, out_channels).to(self.device),
        #                         S=torch.zeros(batch_size, out_channels).to(self.device))
        self.state = LIFDenseNeuronState(self.in_channels, self.out_channels)
        self.fc_layer.weight.data.normal_(mean=0.0, std=self.weight_scale/np.sqrt(in_channels))
        #torch.nn.init.normal_(self.fc_layer.weight.data, mean=0.0, std=self.weight_scale/np.sqrt(nb_inputs))
        self.fc_layer.bias.data.uniform_(-.01, .01)


    def forward(self, Sin_t):
        state = self.state
        U = self.alpha*state.U + (1-self.alpha) * 20 * state.I - state.S.detach() #mem
        I = self.beta*state.I + (1-self.beta) * self.fc_layer(Sin_t) #syn
        # update the neuronal state
        S = smooth_step(U)
        self.state.update(U=U, I=I, S=S)
        #self.NeuronState = self.state
        #state = NeuronState(U=U, I=I, S=S)
        return self.state

    def init_state(self):
        self.state.init()


    def init_mod_weights(self,W):
        self.fc_layer.weight = torch.nn.Parameter(self.fc_layer.weight.data * torch.tensor(W,dtype=float))
        
class LifRecPopulation(nn.Module):
    # NeuronState = namedtuple('NeuronState', ['U', 'I', 'S'])
    def __init__(self, in_channels, out_channels, bias=True, alpha = .9, beta=.85):
        super(LifRecPopulation, self).__init__()
        self.fc_layer = nn.Linear(in_channels, out_channels)
        self.rec_layer = nn.Linear(out_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_scale = 0.2
        self.alpha = alpha
        self.beta = beta
        self.state = LIFDenseNeuronState(self.in_channels, self.out_channels)
        #torch.nn.init.normal_(self.fc_layer.weight.data, mean=0.0, std=self.weight_scale/np.sqrt(nb_inputs))
        self.fc_layer.weight.data.normal_(mean=0.0, std=self.weight_scale/np.sqrt(in_channels))
        self.fc_layer.bias.data.uniform_(-.01, .01)
        #torch.nn.init.normal_(self.rec_layer.weight.data, mean=0.0, std=self.weight_scale/np.sqrt(nb_inputs))
        self.rec_layer.bias.data.uniform_(-.01, .01)
        self.rec_layer.weight.data.fill_(0)


    def forward(self, Sin_t):
        state = self.state
        U = self.alpha*state.U + (1-self.alpha) * 20 * state.I - state.S.detach() #mem
        I = self.beta*state.I + (1 - self.beta) * (self.fc_layer(Sin_t) + self.rec_layer(state.S)) #syn
        # update the neuronal state
        S = smooth_step(U)
        self.state.update(U=U, I=I, S=S)
        #state = NeuronState(U=U, I=I, S=S)
        return self.state

    def init_state(self):
        self.state.init()

    def init_mod_weights(self,W):
        self.fc_layer.weight = torch.nn.Parameter(self.fc_layer.weight.data * torch.tensor(W))

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''
    scale = 100.0
    @staticmethod
    def forward(aux, x):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        aux.save_for_backward(x)
        out = torch.zeros_like(x)

        out[x > 0] = 1.0
        return out

    def backward(aux, grad_output):
        
        #grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SmoothStep.scale*torch.abs(input)+1.0)**2
        return grad

smooth_step = SmoothStep().apply

class OneHiddenModel(nn.Module):

    def __init__(self,in_channels,hidden_channels,out_channels,batch_size,alpha=.9,beta=.85,device='cpu',W=None):
        super(OneHiddenModel, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.device = device
        self.W = W
        self.layer1 = LIFDensePopulation(in_channels=self.in_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer2 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer3 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.out_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,device=device).to(device)

    def forward(self,Sin):
        hidden1 = self.layer1(Sin)
        hidden2 = self.layer2(hidden1.S)
        out = self.layer3(hidden2.S)
        return out

    def init_states(self):
        self.layer1.init_state()
        self.layer2.init_state()
        self.layer3.init_state()

    def init_mod_weights(self,W):
        self.layer2.fc_layer.weight = torch.nn.Parameter((self.layer2.fc_layer.weight.data * torch.tensor(W)).float())

class OneRecHiddenModel(nn.Module):

    def __init__(self,in_channels,hidden_channels,out_channels,batch_size,alpha=.9,beta=.85,device='cpu',W=None):
        super(OneRecHiddenModel, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.device = device
        self.W = W
        self.layer1 = LIFDensePopulation(in_channels=self.in_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer2 = LifRecPopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer3 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.out_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,device=device).to(device)

    def forward(self,Sin):
        hidden1 = self.layer1(Sin)
        hidden2 = self.layer2(hidden1.S)
        out = self.layer3(hidden2.S)
        return out

    def init_states(self):
        self.layer1.init_state()
        self.layer2.init_state()
        self.layer3.init_state()

    def init_mod_weights(self,W):
        self.layer2.fc_layer.weight = torch.nn.Parameter((self.layer2.fc_layer.weight.data * W).to(torch.float32))
        
class ThreeHiddenModel(nn.Module):

    def __init__(self,in_channels,hidden_channels,out_channels,batch_size,alpha=.9,beta=.85,device='cpu',W=None):
        super(ThreeHiddenModel, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.device = device
        self.W = W
        self.layer1 = LIFDensePopulation(in_channels=self.in_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W).to(device)
        self.layer2 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)
        self.layer3 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)
        self.layer4 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)
        self.layer5 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.out_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)

    def forward(self,Sin):
        hidden1 = self.layer1(Sin)
        hidden2 = self.layer2(hidden1.S)
        hidden3 = self.layer3(hidden2.S)
        hidden4 = self.layer4(hidden3.S)
        out = self.layer5(hidden4.S)
        return out

    def init_states(self):
        self.layer1.init_state()
        self.layer2.init_state()
        self.layer3.init_state()
        self.layer4.init_state()
        self.layer5.init_state()

    def init_mod_weights(self,W):
        self.layer1.fc_layer.weight = torch.nn.Parameter(self.layer1.fc_layer.weight.data * torch.Tensor(W))
        self.layer2.fc_layer.weight = torch.nn.Parameter(self.layer2.fc_layer.weight.data * torch.Tensor(W))
        self.layer3.fc_layer.weight = torch.nn.Parameter(self.layer3.fc_layer.weight.data * torch.Tensor(W))
        self.layer4.fc_layer.weight = torch.nn.Parameter(self.layer4.fc_layer.weight.data * torch.Tensor(W))


class NHiddenModel(nn.Module):

    def __init__(self, num_hidden_layers, in_channels,hidden_channels,out_channels,alpha=.9,beta=.85,with_recurrent=True):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        layer_first = LIFDensePopulation(in_channels=self.in_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta)

        hidden_layer_type = LifRecPopulation if with_recurrent else LIFDensePopulation
        hidden_layers = torch.nn.ModuleList([
            hidden_layer_type(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                            alpha=self.alpha,beta=self.beta)
            for ilay in range(num_hidden_layers)    
        ])
        layer_final = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.out_channels,
                                         alpha=self.alpha,beta=self.beta)
        #maybe try last layer non- spiking

        self.layers = torch.nn.ModuleList([
            layer_first,
            *hidden_layers,
            layer_final,
        ])

    def forward(self,Sin):
        spikes = Sin
        for layer in self.layers:
            state = layer(spikes)
            spikes = state.S
        return state

    def init_states(self):
        for layer in self.layers:
            layer.init_state()

    def init_mod_weights(self,W):
        #self.layer1.fc_layer.weight = torch.nn.Parameter(self.layer1.fc_layer.weight.data * torch.Tensor(W))
        num_layers = len(self.layers)
        for ihid in range(1, num_layers-1):
            self.layers[ihid].fc_layer.weight = torch.nn.Parameter(self.layers[ihid].fc_layer.weight.data * W)

class FiveRecHiddenModel(nn.Module):

    def __init__(self,in_channels,hidden_channels,out_channels,batch_size,alpha=.9,beta=.85,device='cpu',W=None):
        super(FiveRecHiddenModel, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.device = device
        self.W = W
        self.layer1 = LIFDensePopulation(in_channels=self.in_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer2 = LifRecPopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer3 = LifRecPopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer4 = LifRecPopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)
        self.layer5 = LifRecPopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)     
        self.layer6 = LifRecPopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W,device=device).to(device)                                                                                                                                                               
        self.layer7= LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.out_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,device=device).to(device)

    def forward(self,Sin):
        hidden1 = self.layer1(Sin)
        hidden2 = self.layer2(hidden1.S)
        hidden3 = self.layer3(hidden2.S)
        hidden4 = self.layer4(hidden3.S)
        hidden5 = self.layer5(hidden4.S)
        hidden6 = self.layer6(hidden5.S)
        out = self.layer7(hidden6.S)
        return out

    def init_states(self):
        self.layer1.init_state()
        self.layer2.init_state()
        self.layer3.init_state()
        self.layer4.init_state()
        self.layer5.init_state()
        self.layer6.init_state()
        self.layer7.init_state()

    def init_mod_weights(self,W):
        self.layer2.fc_layer.weight = torch.nn.Parameter((self.layer2.fc_layer.weight.data * W).to(torch.float32))
        self.layer3.fc_layer.weight = torch.nn.Parameter((self.layer3.fc_layer.weight.data * W).to(torch.float32))
        self.layer4.fc_layer.weight = torch.nn.Parameter((self.layer4.fc_layer.weight.data * W).to(torch.float32))
        self.layer5.fc_layer.weight = torch.nn.Parameter((self.layer5.fc_layer.weight.data * W).to(torch.float32))
        self.layer6.fc_layer.weight = torch.nn.Parameter((self.layer6.fc_layer.weight.data * W).to(torch.float32))