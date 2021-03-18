import torch
import mgplvm as mgp
from torch.distributions import MultivariateNormal, kl_divergence, transform_to, constraints, Normal


def bfa_true_loglik(y, x, sigma):
    m = x.shape[-1]
    n = y.shape[-2]
    kernel = x.transpose(
        -1, -2).matmul(x) + torch.square(sigma)[:, None, None] * torch.eye(m)
    dist = MultivariateNormal(torch.zeros(n, m), covariance_matrix=kernel)
    lp = dist.log_prob(y)
    return lp.sum()


def test_bfa():
    n_samples = 1
    m = 200
    n = 5
    d = 3
    c = torch.randn(n, d)
    sigma = 1E-3
    xtrain = torch.randn(n_samples, d, m)
    ytrain = c.matmul(xtrain) + sigma * torch.randn(n_samples, n, m)
    lik = mgp.likelihoods.Gaussian(n)

    model = mgp.models.Bfa(n, d, m, n_samples, lik)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    for k in range(1500):
        optimizer.zero_grad()
        loglik, kl = model.elbo(ytrain, xtrain)
        loss = -(loglik - kl).sum()
        if k % 500 == 0:
            true_log_prob = bfa_true_loglik(ytrain, xtrain, model.likelihood.sigma)
            xtest = torch.randn(n_samples, d, m)
            ypred, _ = model.predict(xtest, full_cov=False)
            ytest = c.matmul(xtest)
            err = torch.mean(torch.square(ypred - ytest)).item() / torch.mean(
                torch.square(ytest)).item()
            assert (-loss.item() <= true_log_prob.item())
            print(-loss.item(), true_log_prob.item(),
                  torch.mean(torch.square(model.likelihood.sigma)).item(), err)
        loss.backward()
        optimizer.step()
    assert (err < 1e-4)

if __name__ == '__main__':
    test_bfa()
