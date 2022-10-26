import lib.utils
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import GPT2TokenizerFast, GPT2Model

BATCH_SIZE = 64
DECAY_FACTOR = 0.1
GAMMA_0 = -10.0
GAMMA_1 = 10.0

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)

def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    """KL(p||q)"""
    return (
        sigma_q.log() - sigma_p.log()
        + (sigma_p**2 + (mu_p - mu_q)**2)/(2*sigma_q**2)
        - 0.5
    )

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
x = tokenizer('Hello world!',return_tensors='pt')['input_ids'][0][None,:].cuda()
x = x.expand(BATCH_SIZE, -1)

class GPT2(nn.Module):
    """
    GPT2 with two modifications:
    (1) takes one-hot vectors as inputs instead of integers
    (2) predicts the current token instead of the next token
    """
    def __init__(self):
        super().__init__()
        cache_dir = os.path.join(os.environ['DATA_DIR'],
            'huggingface/transformers')
        self.gpt2 = GPT2Model.from_pretrained('gpt2', cache_dir=cache_dir)
    def forward(self, x_onehot):
        h = x_onehot @ self.gpt2.wte.weight
        bos_embedding = self.gpt2.wte.weight[None,None,tokenizer.bos_token_id,:]
        h = torch.cat([bos_embedding.expand(x_onehot.shape[0],-1,-1), h], dim=1)
        h += self.gpt2.wpe.weight[None, :h.shape[1], :]
        h = self.gpt2.drop(h)
        for block in self.gpt2.h:
            h = block(h)[0]
        h = self.gpt2.ln_f(h)
        logits = h @ self.gpt2.wte.weight.T
        return logits[:,:-1,:]

model = GPT2().half().cuda().eval()
lib.utils.print_model(model)

x_onehot = F.one_hot(x, num_classes=len(tokenizer)).double()
logits = model(x_onehot.half()).double()
nll = F.cross_entropy(logits.permute(0,2,1), x, reduction='none').mean(dim=0).sum()
print('Autoregressive NLL:', nll.item())

embedding_matrix = torch.randn((len(tokenizer), 32)).cuda()

scale = DECAY_FACTOR ** torch.arange(x.shape[1]).cuda()[None,:,None]
x_embed = (x_onehot @ embedding_matrix) * scale

def diffusion_vars(t):
    gamma = GAMMA_0 + (t * (GAMMA_1 - GAMMA_0))
    alpha_squared = torch.sigmoid(-gamma)
    sigma_squared = torch.sigmoid(gamma)
    alpha = alpha_squared.sqrt()
    sigma = sigma_squared.sqrt()
    z = (alpha * x_embed) + (sigma * torch.randn_like(x_embed))
    return gamma, alpha_squared, sigma_squared, alpha, sigma, z

def z_log_likelihoods(z, alpha, alpha_squared, sigma_squared):
    # Gaussian p(z|x) for each possible value of x, up to a constant
    z_dot_mu = (z * alpha * scale) @ embedding_matrix.T
    mu_dot_mu = torch.bmm(
        embedding_matrix[:,None,:],
        embedding_matrix[:,:,None]
    )[None,None,:,0,0] * alpha_squared * scale.pow(2)
    return (z_dot_mu - (0.5 * mu_dot_mu)) / sigma_squared

def compute_elbo():
    t = torch.rand(x.shape[0]).cuda()[:,None,None]
    gamma_t, alpha_squared_t, sigma_squared_t, alpha_t, sigma_t, z_t = diffusion_vars(t)
    gamma_0, alpha_squared_0, sigma_squared_0, alpha_0, sigma_0, z_0 = diffusion_vars(torch.zeros_like(t))
    gamma_1, alpha_squared_1, sigma_squared_1, alpha_1, sigma_1, z_1 = diffusion_vars(torch.ones_like(t))

    # ELBO first term: logp(x|z_0)
    logp_x_given_z0 = z_log_likelihoods(z_0, alpha_0, alpha_squared_0, sigma_squared_0)
    reconst_loss = F.cross_entropy(
        logp_x_given_z0.permute(0,2,1), x, reduction='none'
    ).mean(dim=0).sum()

    # ELBO middle term: diffusion loss
    logp_zt_given_x = z_log_likelihoods(z_t, alpha_t, alpha_squared_t, sigma_squared_t)
    z_onehot = F.softmax(logp_zt_given_x, dim=-1)
    p_x_given_z = F.softmax(model(z_onehot.half()).double() + logp_zt_given_x, dim=-1)
    x_pred = (p_x_given_z @ embedding_matrix) * scale
    diffusion_loss = (0.5 * (GAMMA_1 - GAMMA_0) * (-gamma_t).exp() * (x_embed - x_pred).pow(2)).mean(dim=0).sum()

    # ELBO last term: KL( q(z_1|x) || p(z_1) )
    prior_loss = gaussian_kl(
        alpha_1 * x_embed,
        sigma_1,
        torch.tensor(0.).cuda(),
        torch.tensor(1.).cuda()
    ).mean(dim=0).sum()
    elbo = reconst_loss + diffusion_loss + prior_loss

    return elbo

print('Diffusion ELBO (average so far):')
all_elbos = []
for i in range(100_000):
    all_elbos.append(compute_elbo().item())
    if i % 1000 == 999:
        print(np.mean(all_elbos))