# %%
import matplotlib.pyplot as plt
import pyro
import seaborn as sns
import torch
from pyro.infer.mcmc import HMC, MCMC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# depends on your network connection,
# it make take more than a few mintues
mnist = fetch_openml("mnist_784")

x_train, x_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=0.2, random_state=2020,
)


# %%
# create a DecisionTreeClassifier object and fit with training data
clf_dt = DecisionTreeClassifier()
clf_dt.fit(x_train, y_train)


# %%
# create a LogisticRegression object and fit with training data
clf_lr = LogisticRegression()
clf_lr.fit(x_train, y_train)


# %%
# get prediction label using clf_dt and print the accuracy score
y_pred_dt = clf_dt.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"acc_dt: {acc_dt: .2f}")



# %%
# get prediction label using clf_dt and print the accuracy score
y_pred_lr = clf_lr.predict(x_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"acc_lr: {acc_lr: .2f}")


# %%
# label to be used in Monte Carlo method
y_dt = torch.Tensor(y_pred_dt == y_test)
y_lr = torch.Tensor(y_pred_lr == y_test)


# %%
def model():
    # create a variable underlying_p for the Bernoulli distribution
    # where underlying_p is a sample of Uniform distribution from 0 to 1
    # underlying_p = ... 

    # create a hidden Bernoulli distribution with p = underlying_p
    # y_hidden_dist = ... pyro.distributions.Bernoulli(underlying_p)

    # sample the label from the hidden Bernoulli distribution
    underlying_p = pyro.sample(
        "p", pyro.distributions.Uniform(0.0, 1.0)
    )
    y_hidden_dist = pyro.distributions.Bernoulli(underlying_p)
    y_real = pyro.sample("obs", y_hidden_dist)
    return y_real


def conditioned_model(model, y):
    conditioned_model_function = pyro.poutine.condition(
        model, data={"obs": y}
    )
    return conditioned_model_function()


def monte_carlo(y):
    pyro.clear_param_store()

    # create a Simple Hamiltonian Monte Carlo kernel with step_size of 0.1
    hmc_kernel = HMC(conditioned_model, step_size= .1)

    # create a Markov Chain Monte Carlo method with: 
    # the hmc_kernel, 500 samples, and 100 warmup iterations
    mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
    
    mcmc.run(model, y)

    sample_dict = mcmc.get_samples(num_samples=5000)
    plt.figure(figsize=(8, 6))
    sns.distplot(sample_dict["p"].numpy())
    plt.xlabel("Observed probability value")
    plt.ylabel("Observed frequency")
    plt.show()
    mcmc.summary(prob=0.95)

    return sample_dict


# %%
# run the Monte Carlo method with y_dt and save the sample_dict with name simulations_dt
simulations_dt = monte_carlo(y_dt)

#%%
# run the Monte Carlo method with y_dt and save the sample_dict with name simulations_lr
simulations_lr = monte_carlo(y_lr)

# %%
plt.figure(figsize=(8, 6))
sns.distplot(
    simulations_dt["p"].numpy(), label="DecisionTree", color="red"
)
sns.distplot(
    simulations_lr["p"].numpy(),
    label="LogisticRegression",
    color="green",
)
plt.legend()
plt.show()
