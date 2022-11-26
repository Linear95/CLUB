import numpy as np
import math

import torch 
import torch.nn as nn

class CLUBForCategorical(nn.Module): # Update 04/27/2022
    '''
    This class provide a CLUB estimator to calculate MI upper bound between vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    '''
    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels 
        '''
        super().__init__()
        
        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )
            
    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        logits = self.variational_net(inputs)  #[sample_size, label_num]
        
        # log of conditional probability of positive sample pairs
        #positive = - nn.functional.cross_entropy(logits, labels, reduction='none')    
        sample_size, label_num = logits.shape
        
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)     # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )
        
        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)
    
    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)
    

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
   

class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0. Update 11/26/2022
    def __init__(self, x_dim, y_dim, hidden_size=None):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        
        super(CLUBMean, self).__init__()
   
        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
                                       nn.ReLU(),
                                       nn.Linear(int(hidden_size), y_dim))


    def get_mu_logvar(self, x_samples):
        # variance is set to 1, which means logvar=0
        mu = self.p_mu(x_samples)
        return mu, 0
    
    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2.
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2.

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


    
    
    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

    
class NWJ(nn.Module):   
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
                                    
    def forward(self, x_samples, y_samples): 
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))-1.  #shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1) - np.log(sample_size)).exp().mean() 
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


    
class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)



def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples): 
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        negative = log_sum_exp(all_probs + diag_mask,dim=0) - np.log(batch_size-1.) #[nsample]
      
        return (positive - negative).mean()
        
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

    
class VarUB(nn.Module):  #    variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
            
    def forward(self, x_samples, y_samples): #[nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1./2.*(mu**2 + logvar.exp() - 1. - logvar).mean()
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

    
