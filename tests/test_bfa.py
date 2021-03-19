import torch
import mgplvm as mgp

def test_fa():
    n_samples = 2
    m = 200
    n = 5
    d = 3
    c = torch.randn(n, d)
    sigma = 1E-3
    xtrain = torch.randn(n_samples, d, m)
    ytrain = c.matmul(xtrain) + sigma * torch.randn(n_samples, n, m)
    fa = mgp.models.fa(n,d)
    optimizer = torch.optim.Adam(fa.parameters(), lr=0.002)
    for k in range(5000):
        optimizer.zero_grad()
        lp = fa.log_prob(ytrain, xtrain)
        if k % 500 == 0:
            xtest = torch.randn(n_samples, d, m)
            ytest = c.matmul(xtest)
            ypred, _ = fa.predict(xtest, full_cov=False)
            err = torch.mean(torch.square(ypred - ytest)).item() / torch.mean(
                torch.square(ytest)).item()
            print(lp.item(), torch.mean(torch.square(fa.sigma)).item(), err)
        (-lp).backward()
        optimizer.step()
    assert (err < 5e-4)

def test_bfa():
    n_samples = 2
    m = 200
    n = 5
    d = 3
    c = torch.randn(n, d)
    sigma = 1E-3
    xtrain = torch.randn(n_samples, d, m)
    ytrain = c.matmul(xtrain) + sigma * torch.randn(n_samples, n, m)
    bfa = mgp.models.Bfa(n)
    optimizer = torch.optim.Adam(bfa.parameters(), lr=0.001)
    for k in range(100):
        optimizer.zero_grad()
        lp = bfa.log_prob(ytrain, xtrain)
        if k % 50 == 0:
            xtest = torch.randn(n_samples, d, m)
            ytest = c.matmul(xtest)
            ypred, _ = bfa.predict(xtest, ytrain, xtrain, full_cov=False)
            err = torch.mean(torch.square(ypred - ytest)).item() / torch.mean(
                torch.square(ytest)).item()
            print(lp.item(), torch.mean(torch.square(bfa.sigma)).item(), err)
        (-lp).backward()
        optimizer.step()
    assert (err < 5e-4)


def test_bvfa():
    n_samples = 2
    m = 200
    n = 5
    d = 3
    c = torch.randn(n, d)
    sigma = 1E-3
    xtrain = torch.randn(n_samples, d, m)
    ytrain = c.matmul(xtrain) + sigma * torch.randn(n_samples, n, m)
    lik = mgp.likelihoods.Gaussian(n)

    model = mgp.models.Bvfa(n, d, m, n_samples, lik)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for k in range(1000):
        optimizer.zero_grad()
        loglik, kl = model.elbo(ytrain, xtrain)
        loss = -(loglik - kl).sum()
        bfa = mgp.models.Bfa(n, model.likelihood.sigma.data, learn_sigma=False)
        true_log_prob = bfa.log_prob(ytrain, xtrain)
        if k % 200 == 0:
            xtest = torch.randn(n_samples, d, m)
            ytest = c.matmul(xtest)
            ypred, _ = model.predict(xtest, full_cov=False)
            err = torch.mean(torch.square(ypred - ytest)).item() / torch.mean(
                torch.square(ytest)).item()
            assert (-loss.item() <= true_log_prob.item())
            print(-loss.item(), true_log_prob.item(),
                  torch.mean(torch.square(model.likelihood.sigma)).item(), err)
        loss.backward()
        optimizer.step()
    assert (err < 5e-4)


if __name__ == '__main__':
    test_fa()
    test_bfa()
    test_bvfa()
