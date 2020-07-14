import tensorflow as tf

layers = tf.contrib.layers
ds = tf.contrib.distributions
def encoder(images):
    net = layers.relu(2*images-1, 1024)
    net = layers.relu(net, 1024)
    params = layers.linear(net, 512)
    mu, rho = params[:, :256], tf.clip_by_value(params[:, 256:], -1.1, 1.1)
    encoding = ds.NormalWithSoftplusScale(mu, rho)
    return encoding


def decoder(encoding_sample):
    net = layers.linear(encoding_sample, 10)
    return net

def mi(inputs):
    inputs = 2 * inputs - 1
    inputs += tf.random_normal(shape=tf.shape(inputs), seed=1)*0.3
    mu_appr = layers.elu(inputs, 256)
    mu_appr = layers.linear(mu_appr, 256)
    logvar_appr = layers.elu(inputs, 256)
    logvar_appr = layers.linear(logvar_appr, 256)
    logvar_appr = tf.math.tanh(logvar_appr)*2
    
    return mu_appr, logvar_appr