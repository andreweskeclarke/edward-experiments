# http://edwardlib.org/tutorials/supervised-regression


import ipdb
import numpy as np
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt


from edward.models import Normal


def build_toy_dataset(N, w, noise_std=0.1):
  D = len(w)
  x = np.random.randn(N, D)
  y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
  return x, y

N = 40
D = 10
w_true = np.random.randn(D)
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)

# Bayesian regression:
# p(w) ~ N(0, sigma_w^2 * I)
# p(b) ~ N(0, sigma_b^2)
# p(y|x,w,b) ~ N(x'w + b, sigma_y^2) 
# p(Y|x,w,b) ~ Prod[ N(x'w + b, sigma_y^2) ]

# Build the compute graphs matching our Bayesian regression model:
X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w), scale=tf.ones(N))

# Variational approximations - fully factorized w and b graphs:
# This doesn't really count as an approximation, but whatever.
qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

# Run Kullback-Leibler divergence inference
# Minimize KL[q(w) || p(w|X)] where 
#   q(w) ~ Normal
#   p(w|X) = p(X|w)p(w) / p(X)
inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run(n_samples=5, n_iter=250)

# "Criticism" - edward generates a new graph for y_post, swapping out
# w and qw whenever they appear in the graph
# y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(N))
y_post = ed.copy(y, {w: qw, b: qb})

# Some stored edward loss functions
print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))

def visualise(X_data, y_data, w, b, n_samples=10):
  w_samples = w.sample(n_samples)[0].eval()
  b_samples = b.sample(n_samples).eval()
  plt.scatter(X_data[:, 0], y_data)
  inputs = np.linspace(-8, 8, num=400)
  for ns in range(n_samples):
    output = inputs * w_samples[ns] + b_samples[ns]
    plt.plot(inputs, output)

plt.figure()
visualise(X_train, y_train, w, b)
plt.figure()
visualise(X_train, y_train, qw, qb)
plt.show()
