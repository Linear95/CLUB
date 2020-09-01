import tensorflow as tf
import utils
import pdb
import math
import tensorflow.contrib.layers as layers
#import keras.backend as K

def leaky_relu(x, a=0.1):
    return tf.maximum(x, a * x)

def noise(x, phase=True, std=1.0):
    eps = tf.random_normal(tf.shape(x), 0.0, std)
    output = tf.where(phase, x + eps, x)
    return output

class MNISTModel_DANN(object):
    """Simple MNIST domain adaptation model."""
    def __init__(self, options):
        self.reg_disc = options['reg_disc']
        self.reg_con = options['reg_con']
        self.reg_tgt = options['reg_tgt']
        self.lr_g = options['lr_g']
        self.lr_d = options['lr_d']
        self.sample_type = tf.float32
        self.num_labels = options['num_labels']
        self.num_domains = options['num_domains']
        self.num_targets = options['num_targets']
        self.sample_shape = options['sample_shape']
        self.ef_dim = options['ef_dim']
        self.latent_dim = options['latent_dim']
        self.batch_size = options['batch_size']
        self.initializer = tf.contrib.layers.xavier_initializer()
        # self.initializer = tf.truncated_normal_initializer(stddev=0.1)
        self.X = tf.placeholder(tf.as_dtype(self.sample_type), [None] + list(self.sample_shape), name="input_X")
        self.y = tf.placeholder(tf.float32, [None, self.num_labels], name="input_labels")
        self.domains = tf.placeholder(tf.float32, [None, self.num_domains], name="input_domains")
        self.train = tf.placeholder(tf.bool, [], name = 'train')
        self._build_model()
        self._setup_train_ops()

    # def feature_extractor(self, reuse = False):
    #     input_X = utils.normalize_images(self.X)
    #     with tf.variable_scope('feature_extractor_conv1',reuse = reuse):
    #         h_conv1 = layers.conv2d(input_X, self.ef_dim, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_pool1 = layers.max_pool2d(h_conv1, [2, 2], 2, padding='SAME')
            
    #     with tf.variable_scope('feature_extractor_conv2',reuse = reuse):  
    #         h_conv2 = layers.conv2d(h_pool1, self.ef_dim * 2, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_pool2 = layers.max_pool2d(h_conv2, [2, 2], 2, padding='SAME')

    #     with tf.variable_scope('feature_extractor_conv3',reuse = reuse):  
    #         h_conv3 = layers.conv2d(h_pool2, self.ef_dim * 4, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_pool3 = layers.max_pool2d(h_conv3, [2, 2], 2, padding='SAME')
            
    #     with tf.variable_scope('feature_extractor_fc1'):
    #         fc_input = layers.flatten(h_pool3)
    #         fc_1 = layers.fully_connected(inputs=fc_input, num_outputs=self.latent_dim,
    #                                           activation_fn=tf.nn.relu, weights_initializer=self.initializer)  
            
    #         self.features = fc_1
    #         feature_shape = self.features.get_shape()
    #         self.feature_dim = feature_shape[1].value

    #         self.features_src = tf.slice(self.features, [0, 0], [self.batch_size, -1])
    #         self.features_for_prediction =  tf.cond(self.train,  lambda: tf.slice(self.features, [0, 0], [self.batch_size, -1]), lambda: self.features)
    
    # def feature_extractor(self, reuse = False):
    #     input_X = utils.normalize_images(self.X)
    #     with tf.variable_scope('feature_extractor_conv1',reuse = reuse):
    #         h_conv1 = layers.conv2d(input_X, self.ef_dim, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_conv1 = layers.conv2d(h_conv1, self.ef_dim, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_conv1 = layers.max_pool2d(h_conv1, [2, 2], 2, padding='SAME')        
            
    #     with tf.variable_scope('feature_extractor_conv2',reuse = reuse):  
    #         h_conv2 = layers.conv2d(h_conv1, self.ef_dim * 2, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_conv2 = layers.conv2d(h_conv2, self.ef_dim * 2, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_conv2 = layers.max_pool2d(h_conv2, [2, 2], 2, padding='SAME')
            
    #     with tf.variable_scope('feature_extractor_conv3',reuse = reuse):  
    #         h_conv3 = layers.conv2d(h_conv2, self.ef_dim * 4, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_conv3 = layers.conv2d(h_conv3, self.ef_dim * 4, 3, stride=1,
    #                                 activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #         h_conv3 = layers.max_pool2d(h_conv3, [2, 2], 2, padding='SAME')
            
    #     with tf.variable_scope('feature_extractor_fc1'):
    #         # fc_input = tf.nn.dropout(layers.flatten(h_conv3), keep_prob = 0.9)
    #         fc_input = layers.flatten(h_conv3)
    #         fc_1 = layers.fully_connected(inputs=fc_input, num_outputs=self.latent_dim,
    #                                           activation_fn=tf.nn.relu, weights_initializer=self.initializer)

    #         self.features =  fc_1
    #         feature_shape = self.features.get_shape()
    #         self.feature_dim = feature_shape[1].value
    #         self.features_src = tf.slice(self.features, [0, 0], [self.batch_size, -1])
    #         self.features_for_prediction =  tf.cond(self.train,  lambda: tf.slice(self.features, [0, 0], [self.batch_size, -1]), lambda: self.features)
    

    def feature_extractor_c(self, reuse = False):
        training = tf.cond(self.train, lambda: True, lambda: False)
        X = layers.instance_norm(self.X)
        with tf.variable_scope('feature_extractor_c', reuse = reuse):
            h_conv1 = layers.conv2d(self.X, self.ef_dim*3, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv1 = layers.batch_norm(h_conv1, activation_fn=leaky_relu)
            h_conv1 = layers.conv2d(h_conv1, self.ef_dim*3, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv1 = layers.batch_norm(h_conv1, activation_fn=leaky_relu)
            h_conv1 = layers.conv2d(h_conv1, self.ef_dim*3, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv1 = layers.batch_norm(h_conv1, activation_fn=leaky_relu)
            h_conv1 = layers.max_pool2d(h_conv1, 2, 2, padding='SAME')
            h_conv1 = noise(tf.layers.dropout(h_conv1, rate=0.5, training=training), phase=training)
            


            h_conv2 = layers.conv2d(h_conv1, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv2 = layers.batch_norm(h_conv2, activation_fn=leaky_relu)
            h_conv2 = layers.conv2d(h_conv2, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv2 = layers.batch_norm(h_conv2, activation_fn=leaky_relu)
            h_conv2 = layers.conv2d(h_conv2, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv2 = layers.batch_norm(h_conv2, activation_fn=leaky_relu)
            h_conv2 = layers.max_pool2d(h_conv2, 2, 2, padding='SAME')
            h_conv2 = noise(tf.layers.dropout(h_conv2, rate=0.5, training=training), phase=training)



            h_conv3 = layers.conv2d(h_conv2, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv3 = layers.batch_norm(h_conv3, activation_fn=leaky_relu)
            h_conv3 = layers.conv2d(h_conv3, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv3 = layers.batch_norm(h_conv3, activation_fn=leaky_relu)
            h_conv3 = layers.conv2d(h_conv3, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv3 = layers.batch_norm(h_conv3, activation_fn=leaky_relu)
            h_conv3 = tf.reduce_mean(h_conv3, axis=[1, 2])

            self.features_c = h_conv3
            feature_shape = self.features_c.get_shape()
            self.feature_c_dim = feature_shape[1].value
            self.features_c_src = tf.slice(self.features_c, [0, 0], [self.batch_size, -1])
            self.features_c_for_prediction =  tf.cond(self.train,  lambda: tf.slice(self.features_c, [0, 0], [self.batch_size, -1]), lambda: self.features_c)
    
    def feature_extractor_d(self, reuse = False):
        training = tf.cond(self.train, lambda: True, lambda: False)
        X = layers.instance_norm(self.X)
        with tf.variable_scope('feature_extractor_d', reuse = reuse):
            h_conv1 = layers.conv2d(self.X, self.ef_dim*3, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv1 = layers.batch_norm(h_conv1, activation_fn=leaky_relu)
            h_conv1 = layers.conv2d(h_conv1, self.ef_dim*3, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv1 = layers.batch_norm(h_conv1, activation_fn=leaky_relu)
            h_conv1 = layers.conv2d(h_conv1, self.ef_dim*3, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv1 = layers.batch_norm(h_conv1, activation_fn=leaky_relu)
            h_conv1 = layers.max_pool2d(h_conv1, 2, 2, padding='SAME')
            h_conv1 = noise(tf.layers.dropout(h_conv1, rate=0.5, training=training), phase=training)
            


            h_conv2 = layers.conv2d(h_conv1, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv2 = layers.batch_norm(h_conv2, activation_fn=leaky_relu)
            h_conv2 = layers.conv2d(h_conv2, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv2 = layers.batch_norm(h_conv2, activation_fn=leaky_relu)
            h_conv2 = layers.conv2d(h_conv2, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv2 = layers.batch_norm(h_conv2, activation_fn=leaky_relu)
            h_conv2 = layers.max_pool2d(h_conv2, 2, 2, padding='SAME')
            h_conv2 = noise(tf.layers.dropout(h_conv2, rate=0.5, training=training), phase=training)



            h_conv3 = layers.conv2d(h_conv2, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv3 = layers.batch_norm(h_conv3, activation_fn=leaky_relu)
            h_conv3 = layers.conv2d(h_conv3, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv3 = layers.batch_norm(h_conv3, activation_fn=leaky_relu)
            h_conv3 = layers.conv2d(h_conv3, self.ef_dim*6, 3, stride=1, padding='SAME',
                                    activation_fn=None, weights_initializer=self.initializer)
            h_conv3 = layers.batch_norm(h_conv3, activation_fn=leaky_relu)
            h_conv3 = tf.reduce_mean(h_conv3, axis=[1, 2])

            self.features_d = h_conv3
            # self.features_d_src = tf.slice(self.features_d, [0, 0], [self.batch_size, -1])
            # self.features_d_for_prediction =  tf.cond(self.train,  lambda: tf.slice(self.features_d, [0, 0], [self.batch_size, -1]), lambda: self.features_d)
    
    def mi_net(self, input_sample, reuse = False):
        with tf.variable_scope('mi_net', reuse=reuse):
            fc_1 = layers.fully_connected(inputs=input_sample, num_outputs=64, activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            fc_2 = layers.fully_connected(inputs=fc_1, num_outputs=1, activation_fn=None, weights_initializer=self.initializer)
        return fc_2


    def mine(self):
        # tmp_1 = tf.random_shuffle(tf.range(self.batch_size))
        # tmp_2 = tf.random_shuffle(tf.range(self.batch_size))
        # shuffle_d_1 = tf.gather(tf.slice(tf.identity(self.features_d), [0, 0], [self.batch_size, -1]), tmp_1)
        # shuffle_d_2 = tf.gather(tf.slice(tf.identity(self.features_d), [self.batch_size, 0], [self.batch_size, -1]), tmp_2)
        # self.shuffle_d = tf.concat([shuffle_d_1, shuffle_d_2], axis = 0)
        tmp = tf.random_shuffle(tf.range(self.batch_size*2))
        self.shuffle_d = tf.gather(self.features_d, tmp)

        input_0 = tf.concat([self.features_c,self.features_d], axis = -1)
        input_1 = tf.concat([self.features_c,self.shuffle_d], axis = -1)

        T_0 = self.mi_net(input_0)
        T_1 = self.mi_net(input_1, reuse=True)

        E_pos = math.log(2.) - tf.nn.softplus(-T_0)
        E_neg = tf.nn.softplus(-T_1) + T_1 - math.log(2.)

        # grad = tf.gradients(mi_l, [self.features_c, self.features_d, self.shuffle_d])
        # pdb.set_trace()
        # self.penalty = tf.reduce_mean(tf.square(tf.reduce_sum(tf.square(grad))-1.))
        self.bound = tf.reduce_mean(E_pos - E_neg)
        

    def club(self, reuse=False):
        with tf.variable_scope('mi_net', reuse=reuse):
            p_0 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction = layers.fully_connected(inputs=p_0, num_outputs=int(self.features_d.shape[1]), activation_fn=None, weights_initializer=self.initializer)

            p_1 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction_1 = layers.fully_connected(inputs=p_1, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.tanh, weights_initializer=self.initializer)

        mu = prediction
        logvar = prediction_1

        prediction_tile = tf.tile(tf.expand_dims(prediction, dim=1), tf.constant([1, self.batch_size*2, 1], tf.int32))
        features_d_tile = tf.tile(tf.expand_dims(self.features_d, dim=0), tf.constant([self.batch_size*2, 1, 1], tf.int32))
        
        positive = -(mu - self.features_d)**2/2./tf.exp(logvar)
        negative = -tf.reduce_mean((features_d_tile-prediction_tile)**2, 1)/2./tf.exp(logvar)

        # positive = -(prediction-self.features_d)**2
        # negative = -tf.reduce_mean((features_d_tile-prediction_tile)**2, 1)

        self.lld = tf.reduce_mean(tf.reduce_sum(positive, -1))
        self.bound = tf.reduce_mean(tf.reduce_sum(positive, -1)-tf.reduce_sum(negative, -1))

    def club_sample(self, reuse=False):
        with tf.variable_scope('mi_net', reuse=reuse):
            p_0 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction = layers.fully_connected(inputs=p_0, num_outputs=int(self.features_d.shape[1]), activation_fn=None, weights_initializer=self.initializer)

            p_1 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction_1 = layers.fully_connected(inputs=p_1, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.tanh, weights_initializer=self.initializer)

        mu = prediction
        logvar = prediction_1

        tmp = tf.random_shuffle(tf.range(self.batch_size*2))
        self.shuffle_d = tf.gather(self.features_d, tmp)

        positive = -(mu - self.features_d)**2/2./tf.exp(logvar)
        negative = -(mu - self.shuffle_d)**2/2./tf.exp(logvar)

        self.lld = tf.reduce_mean(tf.reduce_sum(positive, -1))
        self.bound = tf.reduce_mean(tf.reduce_sum(positive, -1)-tf.reduce_sum(negative, -1))

    def NWJ(self, reuse=False):
        features_c_tile = tf.tile(tf.expand_dims(self.features_c, dim=0), tf.constant([self.batch_size*2, 1, 1], tf.int32))
        features_d_tile = tf.tile(tf.expand_dims(self.features_d, dim=1), tf.constant([1, self.batch_size*2, 1], tf.int32))
        input_0 = tf.concat([self.features_c, self.features_d], axis = -1)
        input_1 = tf.concat([features_c_tile, features_d_tile], axis = -1)

        T_0 = self.mi_net(input_0)
        T_1 = self.mi_net(input_1, reuse=True) - 1.

        self.bound = tf.reduce_mean(T_0) - tf.reduce_mean(tf.exp(tf.reduce_logsumexp(T_1, 1) - math.log(self.batch_size*2)))

    def VUB(self, reuse=False):
        with tf.variable_scope('mi_net', reuse=reuse):
            p_0 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction = layers.fully_connected(inputs=p_0, num_outputs=int(self.features_d.shape[1]), activation_fn=None, weights_initializer=self.initializer)

            p_1 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction_1 = layers.fully_connected(inputs=p_1, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.tanh, weights_initializer=self.initializer)

        mu = prediction
        logvar = prediction_1

        self.lld = tf.reduce_mean(tf.reduce_sum(-(mu-self.features_d)**2 / tf.exp(logvar) - logvar, -1))
        self.bound = 1. / 2. * tf.reduce_mean(mu**2 + tf.exp(logvar) - 1. - logvar)

    def L1OutUB(self, reuse=False):
        with tf.variable_scope('mi_net', reuse=reuse):
            p_0 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction = layers.fully_connected(inputs=p_0, num_outputs=int(self.features_d.shape[1]), activation_fn=None, weights_initializer=self.initializer)

            p_1 = layers.fully_connected(inputs=self.features_c, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction_1 = layers.fully_connected(inputs=p_1, num_outputs=int(self.features_d.shape[1]), activation_fn=tf.nn.tanh, weights_initializer=self.initializer)

        mu = prediction
        logvar = prediction_1

        positive = tf.reduce_sum(-(mu - self.features_d)**2/2./tf.exp(logvar) - logvar/2., -1)
        
        prediction_tile = tf.tile(tf.expand_dims(prediction, dim=1), tf.constant([1, self.batch_size*2, 1], tf.int32))
        prediction_1_tile = tf.tile(tf.expand_dims(prediction_1, dim=1), tf.constant([1, self.batch_size*2, 1], tf.int32))
        features_d_tile = tf.tile(tf.expand_dims(self.features_d, dim=0), tf.constant([self.batch_size*2, 1, 1], tf.int32))
        
        all_probs = tf.reduce_sum(-(features_d_tile-prediction_tile)**2/2./tf.exp(prediction_1_tile) - prediction_1_tile/2., -1)
        diag_mask = tf.diag([-20.]*self.batch_size*2)

        negative = tf.reduce_logsumexp(all_probs + diag_mask, 0) - math.log(self.batch_size*2 - 1.)
        self.bound = tf.reduce_mean(positive-negative)
        self.lld = tf.reduce_mean(tf.reduce_sum(-(mu - self.features_d)**2/tf.exp(logvar) - logvar, -1))


    def nce(self):
        
        features_c_tile = tf.tile(tf.expand_dims(self.features_c, dim=0), tf.constant([self.batch_size*2, 1, 1], tf.int32))
        features_d_tile = tf.tile(tf.expand_dims(self.features_d, dim=1), tf.constant([1, self.batch_size*2, 1], tf.int32))
        input_0 = tf.concat([self.features_c, self.features_d], axis = -1)
        input_1 = tf.concat([features_c_tile, features_d_tile], axis = -1)

        T_0 = self.mi_net(input_0)
        T_1 = tf.reduce_mean(self.mi_net(input_1, reuse=True), axis=1)

        E_pos = math.log(2.) - tf.nn.softplus(-T_0)
        E_neg = tf.nn.softplus(-T_1) + T_1 - math.log(2.)

        self.bound = tf.reduce_mean(E_pos - E_neg)


    def label_predictor(self):
        # with tf.variable_scope('label_predictor_fc1'):
        #     fc_1 = layers.fully_connected(inputs=self.features_for_prediction, num_outputs=self.latent_dim, 
        #                                     activation_fn=tf.nn.relu, weights_initializer=self.initializer)
        with tf.variable_scope('label_predictor_logits'):
            logits = layers.fully_connected(inputs=self.features_c_for_prediction, num_outputs=self.num_labels, 
                                            activation_fn=None, weights_initializer=self.initializer)
            
        self.y_pred = tf.nn.softmax(logits)
        self.y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.y))
        self.y_acc = utils.predictor_accuracy(self.y_pred, self.y)
        

    def domain_predictor(self, reuse = False):
        with tf.variable_scope('domain_predictor_fc1', reuse = reuse):
            fc_1 = layers.fully_connected(inputs=self.features_d, num_outputs=self.latent_dim, 
                                          activation_fn=tf.nn.relu, weights_initializer=self.initializer)
        with tf.variable_scope('domain_predictor_logits', reuse = reuse):
            self.d_logits = layers.fully_connected(inputs=fc_1, num_outputs=self.num_domains, 
                                              activation_fn=None, weights_initializer=self.initializer)

        
        logits_real = tf.slice(self.d_logits, [0, 0], [self.batch_size, -1])
        logits_fake = tf.slice(self.d_logits, [self.batch_size, 0], [self.batch_size * self.num_targets, -1])
        
        label_real = tf.slice(self.domains, [0, 0], [self.batch_size, -1])        
        label_fake = tf.slice(self.domains, [self.batch_size, 0], [self.batch_size * self.num_targets, -1])
        label_pseudo = tf.ones(label_fake.shape) - label_fake
            
        self.d_pred = tf.nn.sigmoid(self.d_logits)
        real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = label_real))
        fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = label_fake))
        self.d_loss = real_d_loss + self.reg_tgt * fake_d_loss
        self.d_acc = utils.predictor_accuracy(self.d_pred,self.domains)

    # def domain_test(self, reuse=False):
    #     with tf.variable_scope('domain_test_fc1', reuse = reuse):
    #         fc_1 = layers.fully_connected(inputs=self.features, num_outputs=self.latent_dim, 
    #                                       activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #     with tf.variable_scope('domain_test_logits', reuse = reuse):
    #         d_logits = layers.fully_connected(inputs=fc_1, num_outputs=self.num_domains, 
    #                                           activation_fn=None, weights_initializer=self.initializer)
        
    #     logits_real = tf.slice(d_logits, [0, 0], [self.batch_size, -1])
    #     logits_fake = tf.slice(d_logits, [self.batch_size, 0], [self.batch_size * self.num_targets, -1])

    #     self.test_pq = tf.nn.softmax(d_logits)
        
    #     label_real = tf.slice(self.domains, [0, 0], [self.batch_size, -1])        
    #     label_fake = tf.slice(self.domains, [self.batch_size, 0], [self.batch_size * self.num_targets, -1])
        
    #     real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = label_real))
    #     fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = label_fake))
    #     self.d_test = real_d_loss + fake_d_loss

    # def distance(self, a, b):
    #     a_matrix = tf.tile(tf.expand_dims(a, 0), [a.shape[0], 1, 1])
    #     b_matrix = tf.tile(tf.expand_dims(b, 0), [b.shape[0], 1, 1])
    #     b_matrix = tf.transpose(b_matrix, [1,0,2])
    #     distance = K.sqrt(K.maximum(K.sum(K.square(a_matrix - b_matrix), axis=2), K.epsilon()))
    #     return distance

    # def calculate_mask(self, idx, idx_2):
    #     idx_matrix = tf.tile(tf.expand_dims(idx, 0), [idx.shape[0], 1])
    #     idx_2_matrix = tf.tile(tf.expand_dims(idx_2, 0), [idx_2.shape[0], 1])
    #     idx_2_transpose = tf.transpose(idx_2_matrix, [1,0])
    #     mask = tf.cast(tf.equal(idx_matrix, idx_2_transpose), tf.float32)
    #     return mask

    # def contrastive_loss(self, y_true, y_pred, hinge=1.0):
    #     margin = hinge
    #     sqaure_pred = K.square(y_pred)
    #     margin_square = K.square(K.maximum(margin - y_pred, 0))
    #     return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

    def _build_model(self):     
        self.feature_extractor_c()
        self.feature_extractor_d()        
        self.label_predictor()
        self.domain_predictor()
        self.club()
        # self.domain_test()

        # self.src_pred = tf.argmax(tf.slice(self.y_pred, [0, 0], [self.batch_size, -1]), axis=-1)
        # self.distance = self.distance(self.features_src, self.features_src)
        # self.batch_compare = self.calculate_mask(self.src_pred, self.src_pred)
        # self.con_loss = self.contrastive_loss(self.batch_compare, self.distance)

        self.context_loss = self.y_loss + 0.1 * self.bound# + self.reg_con*self.con_loss
        self.domain_loss = self.d_loss

    def _setup_train_ops(self):
        context_vars = utils.vars_from_scopes(['feature_extractor_c', 'label_predictor'])
        domain_vars = utils.vars_from_scopes(['feature_extractor_d', 'domain_predictor'])
        mi_vars = utils.vars_from_scopes(['mi_net'])
        self.domain_test_vars = utils.vars_from_scopes(['domain_test'])
        self.train_context_ops = tf.train.AdamOptimizer(self.lr_g,0.5).minimize(self.context_loss, var_list = context_vars)
        self.train_domain_ops = tf.train.AdamOptimizer(self.lr_d,0.5).minimize(self.domain_loss, var_list = domain_vars)
        self.train_mi_ops = tf.train.AdamOptimizer(self.lr_d,0.5).minimize(-self.lld, var_list = mi_vars)
        # self.test_domain_ops = tf.train.AdamOptimizer(self.lr_d,0.5).minimize(self.d_test, var_list = self.domain_test_vars)
