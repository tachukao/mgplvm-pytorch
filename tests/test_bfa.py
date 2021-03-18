import torch
import mgplvm as mgp
from torch.distributions import LowRankMultivariateNormal


def bfa_true_loglik(y, x, sigma):
    m = x.shape[-1]
    d = x.shape[-2]
    n = y.shape[-2]
    cov_factor = x.transpose(-1, -2)
    cov_diag = torch.square(sigma)[:, None] * torch.ones(m)
    dist = LowRankMultivariateNormal(loc=torch.zeros(n, m),
                                     cov_factor=cov_factor,
                                     cov_diag=cov_diag)
    lp = dist.log_prob(y)
    return lp.sum()


def bfa_true_prediction(xstar, y, x, sigma, full_cov=False):
    m = x.shape[-1]
    d = x.shape[-2]
    n = y.shape[-2]
    cov_factor = x.transpose(-1, -2)
    cov_diag = torch.square(sigma)[:, None] * torch.ones(m)
    dist = LowRankMultivariateNormal(loc=torch.zeros(n, m),
                                     cov_factor=cov_factor,
                                     cov_diag=cov_diag)
    prec = dist.precision_matrix
    l = torch.cholesky(prec, upper=False)
    xl = x.matmul(l)
    mu = xstar.transpose(-1, -2).matmul(
        xl.matmul(l.transpose(-1, -2)).matmul(y[..., None])).squeeze(-1)
    if not full_cov:
        return mu, torch.square(xstar).sum(-2) - torch.square(
            xstar.transpose(-1, -2).matmul(xl)).sum(-1)
    else:
        z = torch.eye(m) - xl.matmul(xl.transpose(-1, 2))
        return mu, xstar.transpose(-1, -2).matmul(z).matmul(xstar)


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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for k in range(1500):
        optimizer.zero_grad()
        loglik, kl = model.elbo(ytrain, xtrain)
        loss = -(loglik - kl).sum()
        true_log_prob = bfa_true_loglik(ytrain, xtrain, model.likelihood.sigma)
        if k % 100 == 0:
            xtest = torch.randn(n_samples, d, m)
            #ypred, _ = model.predict(xtest, full_cov=False)
            ypred, _ = bfa_true_prediction(xtest,
                                           ytrain,
                                           xtrain,
                                           model.likelihood.sigma,
                                           full_cov=False)
            ytest = c.matmul(xtest)
            err = torch.mean(torch.square(ypred - ytest)).item() / torch.mean(
                torch.square(ytest)).item()
            assert (-loss.item() <= true_log_prob.item())
            print(-loss.item(), true_log_prob.item(),
                  torch.mean(torch.square(model.likelihood.sigma)).item(), err)
        (-true_log_prob).backward()
        optimizer.step()
    #assert (err < 1e-4)


if __name__ == '__main__':
    test_bfa()
