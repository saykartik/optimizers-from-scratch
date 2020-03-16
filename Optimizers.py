#############################################################################################################
# To structure our code we borrow ideas from the Pytorch way of writing optimizers, but don't over-engineer
# Every optimizer has a step function which will perform one weight update.
# The optimizer object is aware of state of all necessary variables, the optimization algo specific parameters
# In every step, the optimizer updates its state, consisting of the weights and associated grad memories
# The optimizer also gets 'outside' feedback of gradients computed by the Backprop from the model loss function
# Ex: https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py
# Optimizer Theory and Notation Inspiration - https://ruder.io/optimizing-gradient-descent/index.html#adam
#############################################################################################################

from abc import ABC
import numpy as np


# Define an interface for Optimizer
class Optimizer(ABC):
    def step(self):
        pass


# We show Adam first so we can provide some detailed commentary on some deceptively simple looking code.
# Note that this code takes extensive advantage of mutable objects.
# Dictionaries and numpy arrays assigned and edited later propagate edits back to the source from whence it came.
# ultimately this mutability doesnt work everywhere, and we point it out and patch it up.
class Adam(Optimizer):
    def __init__(self, lr=0.01, beta_1=0.9, beta_2=0.999, e=10 ** (-8)):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.e = e

        # Dont put anything in state just yet.
        self.state = {}

    def step(self):
        state = self.state
        # We can't do this in the constructor because we need to know the size of weights and grads
        # we know that just before calling step, weights must have been set in the state.
        if 'step' not in state:
            self.state['step'] = 0
            state['m_t'] = np.zeros_like(state['weights'])
            state['v_t'] = np.zeros_like(state['weights'])

        # All notation gets easier when i dont have to keep saying state again.
        # momentum and gradient squared exponential average
        m_t, v_t, g_t, w_t = state['m_t'], state['v_t'], state['grads'], state['weights']
        m_t = self.beta_1 * m_t + (1 - self.beta_1) * g_t
        v_t = self.beta_2 * v_t + (1 - self.beta_2) * np.power(g_t, 2)

        # Bias Correction
        self.state['step'] += 1
        m_t_hat = np.divide(m_t, 1 - self.beta_1 ** self.state['step'])
        v_t_hat = np.divide(v_t, 1 - self.beta_2 ** self.state['step'])

        # Final weight update - Note that this will get reflected in the Model weights which called step
        w_t -= self.lr * np.divide(m_t_hat, np.sqrt(v_t_hat) + self.e)

        # We can skip this step, if we had used np.multiply(x,y,out=..) which will not allocate fresh arrays
        # We dont do this here because then we will have to write really ugly out=.. and np.* expressions everywhere
        # to prevent ugliness we could do define functions which default out etc... but we are getting way ahead
        # of ourselves here
        state['m_t'], state['v_t'], state['weights'] = m_t, v_t, w_t


class GradientDescent(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.state = {'step': 0}

    def step(self):
        w_t, g_t = self.state['weights'], self.state['grads']  # Kinda useless, but keep notation consistent
        w_t -= self.lr * g_t
        self.state['step'] += 1  # We dont need this here. But we'll need it for other optimizers
        self.state['w_t'] = w_t  # Bah.. i told you this was not worth it.


class SGDM(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.state = {}

    def step(self):
        state = self.state
        if 'step' not in state:
            self.state['step'] = 0
            state['v_t'] = np.zeros_like(state['weights'])

        self.state['step'] += 1
        v_t, w_t, g_t = state['v_t'], state['weights'], state['grads']
        v_t = self.gamma * v_t + self.lr * g_t
        w_t -= v_t

        state['v_t'], state['weights'] = v_t, w_t


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, e=10 ** (-8)):
        self.lr = lr
        self.e = e

        # Dont put anything in state just yet.
        self.state = {}

    def step(self):
        state = self.state
        if 'step' not in state:
            self.state['step'] = 0
            state['G'] = np.zeros_like(state['weights'])

        G, g_t, w_t = state['G'], state['grads'], state['weights']
        # Cumulative sum of squared gradients
        G += np.power(g_t, 2)

        # Final Weight update
        w_t -= self.lr / (np.sqrt(G + self.e)) * g_t

        state['G'], state['weights'] = G, w_t


class Adadelta(Optimizer):
    def __init__(self, gamma=0.9, e=10 ** (-8)):
        # Adadelta has no learning rate!
        self.e = e
        self.gamma = gamma

        # Dont put anything in state just yet.
        self.state = {}

    def step(self):
        state = self.state
        if 'step' not in state:
            self.state['step'] = 0
            state['Eg2'] = np.zeros_like(state['weights'])  # Exp Avg of Squared Gradients
            state['Edw2'] = np.zeros_like(state['weights'])  # Exp Avg of Squared Weight updates

        self.state['step'] += 1

        Eg2, Edw2, w_t, g_t = state['Eg2'], state['Edw2'], state['weights'], state['grads']

        Eg2 = self.gamma * Eg2 + (1 - self.gamma) * np.power(g_t, 2)  # Exp average od squared Gradients

        RMS_g_t = np.sqrt(Eg2 + self.e)
        RMS_dw_tm1 = np.sqrt(Edw2 + self.e)  # approximate from previous time-step. Will be near zero first time
        dw = RMS_dw_tm1 / RMS_g_t * g_t  # Weight Update value

        # This will be used only in the next step
        Edw2 = self.gamma * Edw2 + (1 - self.gamma) * np.power(dw, 2)  # Exp Average of Squared weight updates

        # Finally update the weights
        w_t -= dw

        state['Eg2'], state['Edw2'], state['weights'] = Eg2, Edw2, w_t


class RMSProp(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9, e=10 ** (-8)):
        self.lr = lr
        self.e = e
        self.gamma = gamma

        # Dont put anything in state just yet.
        self.state = {}

    def step(self):
        state = self.state
        if 'step' not in state:
            self.state['step'] = 0
            state['Eg2'] = np.zeros_like(state['weights'])  # Exp Avg of Squared Gradients

        self.state['step'] += 1

        Eg2, w_t, g_t = state['Eg2'], state['weights'], state['grads']

        Eg2 = self.gamma * Eg2 + (1 - self.gamma) * np.power(g_t, 2)  # Exp average od squared Gradients
        RMS_g_t = np.sqrt(Eg2 + self.e)

        # Finally update the weights
        w_t -= self.lr / RMS_g_t * g_t

        state['Eg2'], state['weights'] = Eg2, w_t
