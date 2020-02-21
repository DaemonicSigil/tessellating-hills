import torch
import numpy as np
import matplotlib.pyplot as plt

DIMS = 16   # number of dimensions that xn has
WSUM = 5    # number of waves added together to make a splotch
EPSILON = 0.0025 # rate at which xn controlls splotch strength
TRAIN_TIME = 5000 # number of iterations to train for
LEARN_RATE = 0.2   # learning rate

torch.random.manual_seed(1729)

# knlist and k0list are integers, so the wavesum is periodic
knlist = torch.randint(-2, 3, (DIMS, WSUM, DIMS)) # wavenumbers : list (controlling dim, wave id, k component)
k0list = torch.randint(-2, 3, (DIMS, WSUM))       # the x0 component of wavenumber : list (controlling dim, wave id)
slist = torch.randn((DIMS, WSUM))                # sin coefficients for a particular wave : list(controlling dim, wave id)
clist = torch.randn((DIMS, WSUM))                # cos coefficients for a particular wave : list (controlling dim, wave id)

# initialize x0, xn
x0 = torch.zeros(1, requires_grad=True)
xn = torch.zeros(DIMS, requires_grad=True)

# numpy arrays for plotting:
x0_hist = np.zeros((TRAIN_TIME,))
xn_hist = np.zeros((TRAIN_TIME, DIMS))

# train:
for t in range(TRAIN_TIME):
    ### model: 
    wavesum = torch.sum(knlist*xn, dim=2) + k0list*x0
    splotch_n = torch.sum(
            (slist*torch.sin(wavesum)) + (clist*torch.cos(wavesum)),
            dim=1)
    foreground_loss = EPSILON * torch.sum(xn * splotch_n)
    loss = foreground_loss - x0
    ###
    print(t)
    loss.backward()
    with torch.no_grad():
        # constant step size gradient descent, with some noise thrown in
        vlen = torch.sqrt(x0.grad*x0.grad + torch.sum(xn.grad*xn.grad))
        x0 -= LEARN_RATE*(x0.grad/vlen + torch.randn(1)/np.sqrt(1.+DIMS))
        xn -= LEARN_RATE*(xn.grad/vlen + torch.randn(DIMS)/np.sqrt(1.+DIMS))
    x0.grad.zero_()
    xn.grad.zero_()
    x0_hist[t] = x0.detach().numpy()
    xn_hist[t] = xn.detach().numpy()

plt.plot(x0_hist)
plt.xlabel('number of steps')
plt.ylabel('x0')
plt.show()
for d in range(DIMS):
    plt.plot(xn_hist[:,d])
plt.xlabel('number of training steps')
plt.ylabel('xn')
plt.show()
