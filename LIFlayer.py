import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
NeuronState = namedtuple('NeuronState', ['U', 'I', 'S'])
class LIFDensePopulation(nn.Module):
    # NeuronState = namedtuple('NeuronState', ['U', 'I', 'S'])
    def __init__(self, in_channels, out_channels, bias=True, alpha = .9, beta=.85, batch_size=10,W=None,device='cpu'):
        super(LIFDensePopulation, self).__init__()
        self.fc_layer = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.device = device
        self.weight_scale = 0.2
        self.alpha = alpha
        self.beta = beta
        self.state = NeuronState(U=torch.zeros(batch_size, out_channels).to(self.device),
                                 I=torch.zeros(batch_size, out_channels).to(self.device),
                                 S=torch.zeros(batch_size, out_channels).to(self.device))
        self.NeuronState = self.state
        self.fc_layer.weight.data.normal_(mean=0.0, std=self.weight_scale/np.sqrt(self.in_channels))
        #torch.nn.init.normal_(self.fc_layer.weight.data, mean=0.0, std=self.weight_scale/np.sqrt(nb_inputs))
        self.fc_layer.bias.data.uniform_(-.01, .01)


    def forward(self, Sin_t):
        state = self.state
        U = self.alpha*state.U + state.I - state.S #mem
        I = self.beta*state.I + self.fc_layer(Sin_t) #syn
        # update the neuronal state
        S = smooth_step(U)
        self.state = NeuronState(U=U, I=I, S=S)
        self.NeuronState = self.state
        #state = NeuronState(U=U, I=I, S=S)
        return self.state

    def init_state(self):

        out_channels = self.out_channels
        self.state = NeuronState(U=torch.zeros(self.batch_size, out_channels,device=self.device),
                                 I=torch.zeros(self.batch_size, out_channels,device=self.device),
                                 S=torch.zeros(self.batch_size, out_channels,device=self.device))
        self.NeuronState = self.state

    def init_mod_weights(self,W):
        self.fc_layer.weight = torch.nn.Parameter(self.fc_layer.weight.data * torch.Tensor(W))

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
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size,W=W).to(device)
        self.layer2 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.out_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)

    def forward(self,Sin):
        hidden = self.layer1(Sin)
        out = self.layer2(hidden.S)
        return out

    def init_states(self):
        self.layer1.init_state()
        self.layer2.init_state()

    def init_mod_weights(self,W):
        self.layer1.fc_layer.weight = torch.nn.Parameter(self.layer1.fc_layer.weight.data * torch.Tensor(W))

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


class FiveHiddenModel(nn.Module):

    def __init__(self,in_channels,hidden_channels,out_channels,batch_size,alpha=.9,beta=.85,device='cpu',W=None):
        super(FiveHiddenModel, self).__init__()

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
        self.layer5 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                        alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)
        self.layer6 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.hidden_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)
        self.layer7 = LIFDensePopulation(in_channels=self.hidden_channels,out_channels=self.out_channels,
                                         alpha=self.alpha,beta=self.beta,batch_size=self.batch_size).to(device)
        #maybe try last layer non- spiking

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
        #self.layer1.fc_layer.weight = torch.nn.Parameter(self.layer1.fc_layer.weight.data * torch.Tensor(W))
        self.layer2.fc_layer.weight = torch.nn.Parameter(self.layer2.fc_layer.weight.data * torch.Tensor(W))
        self.layer3.fc_layer.weight = torch.nn.Parameter(self.layer3.fc_layer.weight.data * torch.Tensor(W))
        self.layer4.fc_layer.weight = torch.nn.Parameter(self.layer4.fc_layer.weight.data * torch.Tensor(W))
        self.layer5.fc_layer.weight = torch.nn.Parameter(self.layer5.fc_layer.weight.data * torch.Tensor(W))
        self.layer6.fc_layer.weight = torch.nn.Parameter(self.layer6.fc_layer.weight.data * torch.Tensor(W))
