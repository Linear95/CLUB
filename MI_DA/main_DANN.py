import os
import pdb
import argparse

import numpy as np
#from sklearn.manifold import TSNE
#import scipy.io

import tensorflow as tf
import tensorflow.contrib.slim as slim

from MNISTModel_DANN import MNISTModel_DANN
import imageloader as dataloader
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25) 

parser = argparse.ArgumentParser(description="Domain Adaptation Training")
parser.add_argument("--data_path", type=str, default="/path/to/dataset/folder/", help="path to dataset folder")
parser.add_argument("--save_path", type=str, default="/path/to/save/dir/", help="path to save experiment output")
parser.add_argument("--source", type=str, default="mnist", help="specify source dataset")
parser.add_argument("--target", type=str, default="svhn", help="specify target dataset")

args = parser.parse_args()

data_path = args.data_path
save_path = args.save_path
batch_size = 64
num_steps = 15000
epsilon = 0.5
M = 0.1
num_test_steps = 5000
valid_steps = 100
datasets = dataloader.load_datasets(data_path,{args.source:1,args.target:1})

# datasets = dataloader.normalize_dataset(datasets)
sources = {args.source:1}
targets = {args.target:1}
description = utils.description(sources, targets)
source_train, source_valid, source_test, target_train, target_valid, target_test = dataloader.source_target(datasets, sources, targets, unify_source = True)

options = {}
options['sample_shape'] = (28,28,3)
options['num_domains'] = 2
options['num_targets'] = 1
options['num_labels'] = 10
options['batch_size'] = batch_size
options['G_iter'] = 1
options['D_iter'] = 1
options['ef_dim'] = 32
options['latent_dim'] = 128
options['t_idx'] = np.argmax(target_test['domains'][0])
options['source_num'] = batch_size
options['target_num'] = batch_size
options['reg_disc'] = 0.1
options['reg_con'] = 0.1
options['lr_g'] = 0.001
options['lr_d'] = 0.001
options['reg_tgt'] = 1.0
description = utils.description(sources, targets)
description = description + '_DANN_' + str(options['reg_disc'])


tf.reset_default_graph()  
graph = tf.get_default_graph()
model = MNISTModel_DANN(options)
sess =  tf.Session(graph = graph, config=tf.ConfigProto(gpu_options=gpu_options)) 
tf.global_variables_initializer().run(session = sess)

record = []

gen_source_batch = utils.batch_generator([source_train['images'], 
                                          source_train['labels'], 
                                          source_train['domains']], batch_size)

gen_target_batch = utils.batch_generator([target_train['images'], 
                                          target_train['labels'], 
                                          target_train['domains']], batch_size)

gen_source_batch_valid = utils.batch_generator([np.concatenate([source_valid['images'], source_test['images']]),
                                               np.concatenate([source_valid['labels'], source_test['labels']]), 
                                               np.concatenate([source_valid['domains'], source_test['domains']])],
                                               batch_size)

gen_target_batch_valid = utils.batch_generator([np.concatenate([target_valid['images'], target_test['images']]),
                                               np.concatenate([target_valid['labels'], target_test['labels']]), 
                                               np.concatenate([target_valid['domains'], target_test['domains']])], 
                                               batch_size)

source_data_valid = np.concatenate([source_valid['images'], source_test['images']])
target_data_valid = np.concatenate([target_valid['images'], target_test['images']])
source_label_valid = np.concatenate([source_valid['labels'], source_test['labels']])

#save_path = './Result/' + description + '/'
save_path = os.path.join(save_path, description)
if not os.path.exists(save_path):
    os.mkdir(save_path)

def compute_MMD(H_fake, H_real, sigma_range=[5]):

    min_len = min([len(H_real),len(H_fake)])
    h_real = H_real[:min_len]
    h_fake = H_fake[:min_len]

    dividend = 1
    dist_x, dist_y = h_fake/dividend, h_real/dividend
    x_sq = np.expand_dims(np.sum(dist_x**2, axis=1), 1)   #  64*1
    y_sq = np.expand_dims(np.sum(dist_y**2, axis=1), 1)    #  64*1
    dist_x_T = np.transpose(dist_x)
    dist_y_T = np.transpose(dist_y)
    x_sq_T = np.transpose(x_sq)
    y_sq_T = np.transpose(y_sq)

    tempxx = -2*np.matmul(dist_x,dist_x_T) + x_sq + x_sq_T  # (xi -xj)**2
    tempxy = -2*np.matmul(dist_x,dist_y_T) + x_sq + y_sq_T  # (xi -yj)**2
    tempyy = -2*np.matmul(dist_y,dist_y_T) + y_sq + y_sq_T  # (yi -yj)**2


    for sigma in sigma_range:
        kxx, kxy, kyy = 0, 0, 0
        kxx += np.mean(np.exp(-tempxx/2/(sigma**2)))
        kxy += np.mean(np.exp(-tempxy/2/(sigma**2)))
        kyy += np.mean(np.exp(-tempyy/2/(sigma**2)))

    gan_cost_g = np.sqrt(kxx + kyy - 2*kxy)
    return gan_cost_g

best_valid = -1
best_acc = -1

best_src_acc = -1
best_src = -1

best_bound_acc = -1
best_bound = 100000

best_iw_acc = -1
best_iw = 100000

best_ben_acc = -1
best_ben = 100000

output_file = save_path+'acc.txt'
print('Training...')
with open(output_file, 'w') as fout:
    for i in range(1, num_steps + 1):          
        # Adaptation param and learning rate schedule as described in the paper
        
        X0, y0, d0 = gen_source_batch.next()
        X1, y1, d1 = gen_target_batch.next()
        
        X = np.concatenate([X0, X1], axis = 0)
        d = np.concatenate([d0, d1], axis = 0)

        for j in range(options['D_iter']):
            # Update Adversary
            _, mi_loss = \
                sess.run([model.train_mi_ops, model.bound],
                         feed_dict={model.X:X, model.train: True})

        for j in range(options['G_iter']):
            # Update Feature Extractor & Lable Predictor np.array([0,0,1,1]).astype('float32')
            _, tploss, tp_acc = \
                sess.run([model.train_context_ops, model.y_loss, model.y_acc],
                         feed_dict={model.X: X, model.y: y0, model.train: True})

        for j in range(options['G_iter']):
            # Update Feature Extractor & Lable Predictor np.array([0,0,1,1]).astype('float32')
            _, td_loss, td_acc = \
                sess.run([model.train_domain_ops, model.d_loss, model.d_acc],
                         feed_dict={model.X: X, model.domains: d, model.train: True})

        if i % 10 == 0:
            print '%s iter %d  mi_loss: %.4f  d_loss: %.4f  p_acc: %.4f' % \
                    (description, i, mi_loss, td_loss, tp_acc)
            
        if i % valid_steps == 0:
            # Calculate bound
            # init_new_vars_op = tf.initialize_variables(model.domain_test_vars)
            # sess.run(init_new_vars_op)

            # for s in range(num_test_steps):
            #     X0_test, y0_test, d0_test = gen_source_batch.next()
            #     X1_test, y1_test, d1_test = gen_target_batch.next()
        
            #     X_test = np.concatenate([X0_test, X1_test], axis = 0)
            #     d_test = np.concatenate([d0_test, d1_test], axis = 0)
                
            #     _ = sess.run(model.test_domain_ops, feed_dict={model.X:X_test, 
            #                                                    model.domains: d_test, model.train: False})

            # source_pq = utils.get_data_pq(sess, model, source_data_valid)
            # target_pq = utils.get_data_pq(sess, model, target_data_valid)

            # st_ratio = float(source_train['images'].shape[0]) / target_train['images'].shape[0]

            # src_qp = source_pq[:,1] / source_pq[:,0] * st_ratio
            # tgt_qp = target_pq[:,1] / target_pq[:,0] * st_ratio

            # w_source_pq = np.copy(src_qp)
            # w_source_pq[source_pq[:,0]<epsilon]=1

            # source_y_loss = utils.get_y_loss(sess, model, source_data_valid, source_label_valid)

            # M = source_y_loss.max()

            # beta = - np.cov(w_source_pq, (w_source_pq*source_y_loss))[0,1] / np.var(w_source_pq)


            # source_feature_valid = utils.get_feature(sess, model, source_data_valid)
            # target_feature_valid = utils.get_feature(sess, model, target_data_valid)
            # MMD_loss = compute_MMD(source_feature_valid, target_feature_valid)
            # E_s = np.mean(source_y_loss)
            # ben_david = E_s+MMD_loss

            # E_ps = np.mean(w_source_pq * source_y_loss)
            # iw = np.mean(src_qp * source_y_loss)
            # d_supp = np.mean(np.logical_and((tgt_qp>=1),(target_pq[:,0]<epsilon)))-np.mean(np.logical_and((src_qp>=1),(source_pq[:,0]<epsilon)))
            
            # bound = E_ps + M * d_supp# + beta * np.mean(w_source_pq) - beta
            # print 'bound: %.4f  E src: %.4f  M: %.4f  d supp: %.4f  iw: %.4f  E_s: %.4f  MMD: %.4f' %(bound, E_ps, M, d_supp, iw, E_s, MMD_loss)

            # source_train_pred = utils.get_data_pred(sess, model, 'y', source_train['images'], source_train['labels'])
            # source_train_acc = utils.get_acc(source_train_pred, source_train['labels'])
            
            source_valid_pred = utils.get_data_pred(sess, model, 'y', source_valid['images'], source_valid['labels'])
            source_valid_acc = utils.get_acc(source_valid_pred, source_valid['labels'])

            target_valid_pred = utils.get_data_pred(sess, model, 'y', target_valid['images'], target_valid['labels'])
            target_valid_acc = utils.get_acc(target_valid_pred, target_valid['labels'])
            
            # source_test_pred = utils.get_data_pred(sess, model, 'y', source_test['images'], source_test['labels'])
            # source_test_acc = utils.get_acc(source_test_pred, source_test['labels'])
            
            target_test_pred = utils.get_data_pred(sess, model, 'y', target_test['images'], target_test['labels'])
            target_test_acc = utils.get_acc(target_test_pred, target_test['labels'])
            
            if target_valid_acc > best_valid:
                best_params = utils.get_params(sess)
                best_valid = target_valid_acc
                best_acc = target_test_acc      

            labd = np.concatenate((source_test['domains'], target_test['domains']), axis = 0)
            print 'src valid: %.4f tgt valid: %.4f tgt test: %.4f best: %.4f ' % \
                    (source_valid_acc, target_valid_acc, target_test_acc, best_acc) 

            acc_store = '%.4f, %.4f, %.4f, %.4f \n'%(source_valid_acc, target_valid_acc, target_test_acc, best_acc)
            fout.write(acc_store)
