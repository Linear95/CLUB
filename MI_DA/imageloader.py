import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
from utils import to_one_hot
import pdb
import cv2
import os
#from scipy.misc import imsave

def load_datasets(data_dir = './', sets={'mnist':1, 'svhn':1, 'mnistm':1, 'usps':1}):
    datasets = {}
    for key in sets.keys():
        datasets[key] = {}
    if sets.has_key('mnist'):
        mnist = input_data.read_data_sets(data_dir + 'MNIST_data', one_hot=True)  
        mnist_train = (mnist.train.images.reshape(55000, 28, 28, 1) * 255).astype(np.uint8)
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        
        # mnist_inv = mnist_train * (-1) + 255
        # mnist_train = np.concatenate([mnist_train, mnist_inv])
        mnist_test = (mnist.test.images.reshape(10000, 28, 28, 1) * 255).astype(np.uint8)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_valid = (mnist.validation.images.reshape(5000, 28, 28, 1) * 255).astype(np.uint8)
        mnist_valid = np.concatenate([mnist_valid, mnist_valid, mnist_valid], 3)
        # datasets['mnist']['train'] = {'images': mnist_train, 'labels': np.concatenate([mnist.train.labels, mnist.train.labels])}
        datasets['mnist']['train'] = {'images': mnist_train, 'labels': mnist.train.labels}
        datasets['mnist']['test'] = {'images': mnist_test, 'labels': mnist.test.labels}
        datasets['mnist']['valid'] = {'images': mnist_valid, 'labels': mnist.validation.labels}


    if sets.has_key('mnist32'):
        mnist = input_data.read_data_sets(data_dir + 'MNIST_data', one_hot=True)  
        mnist_train = (mnist.train.images.reshape(55000, 28, 28, 1) * 255).astype(np.uint8)
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = (mnist.test.images.reshape(10000, 28, 28, 1) * 255).astype(np.uint8)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_valid = (mnist.validation.images.reshape(5000, 28, 28, 1) * 255).astype(np.uint8)
        mnist_valid = np.concatenate([mnist_valid, mnist_valid, mnist_valid], 3)
        
        mnist_train = [np.expand_dims(cv2.resize(x, dsize=(32,32)), 0) for x in mnist_train]
        mnist_train = np.concatenate(mnist_train)
        mnist_test = [np.expand_dims(cv2.resize(x, dsize=(32,32)), 0) for x in mnist_test]
        mnist_test = np.concatenate(mnist_test)
        mnist_valid = [np.expand_dims(cv2.resize(x, dsize=(32,32)), 0) for x in mnist_valid]
        mnist_valid = np.concatenate(mnist_valid)
        
        datasets['mnist32']['train'] = {'images': mnist_train, 'labels': mnist.train.labels}
        datasets['mnist32']['test'] = {'images': mnist_test, 'labels': mnist.test.labels}
        datasets['mnist32']['valid'] = {'images': mnist_valid, 'labels': mnist.validation.labels}      
    
    # if sets.has_key('svhn'):
    #     svhn = scipy.io.loadmat(data_dir + 'SVHN/svhn.mat')
    #     svhn_train = svhn['train'].astype(np.uint8)
    #     svhn_labtrain = svhn['labtrain'].astype(np.int32)
    #     svhn_valid = svhn['val'].astype(np.uint8)
    #     svhn_labval= svhn['labval'].astype(np.int32)
    #     svhn_test = svhn['test'].astype(np.uint8)
    #     svhn_labtest =svhn['labtest'].astype(np.int32)
    #     datasets['svhn']['train'] = {'images': svhn_train, 'labels': svhn_labtrain}
    #     datasets['svhn']['test'] = {'images': svhn_test, 'labels': svhn_labtest}
    #     datasets['svhn']['valid'] = {'images': svhn_valid, 'labels': svhn_labval}

    if sets.has_key('svhn'):
        svhn_train = scipy.io.loadmat(data_dir + 'SVHN/train_32x32.mat')
        svhn_train_data = svhn_train['X'].transpose((3,0,1,2)).astype(np.uint8)
        
        svhn_train_label = svhn_train['y'] + 1
        svhn_train_label[svhn_train_label > 10] = 1
        svhn_train_label = to_one_hot(svhn_train_label)
        
        svhn_valid_data = svhn_train_data[-5000:]
        svhn_train_data = svhn_train_data[:-5000]
        
        svhn_valid_label = svhn_train_label[-5000:]
        svhn_train_label = svhn_train_label[:-5000]
        
        svhn_test = scipy.io.loadmat(data_dir + 'SVHN/test_32x32.mat')
        svhn_test_data = svhn_test['X'].transpose((3,0,1,2)).astype(np.uint8)
        
        svhn_test_label = svhn_test['y'] + 1
        svhn_test_label[svhn_test_label > 10] = 1
        svhn_test_label = to_one_hot(svhn_test_label)
        
        # svhn_train_data = [np.expand_dims(cv2.resize(x, dsize=(28,28)), 0) for x in svhn_train_data]
        # svhn_train_data = np.concatenate(svhn_train_data)
        # svhn_test_data = [np.expand_dims(cv2.resize(x, dsize=(28,28)), 0) for x in svhn_test_data]
        # svhn_test_data = np.concatenate(svhn_test_data)
        # svhn_valid_data = [np.expand_dims(cv2.resize(x, dsize=(28,28)), 0) for x in svhn_valid_data]
        # svhn_valid_data = np.concatenate(svhn_valid_data)

        svhn_train_data = svhn_train_data[:,2:30,2:30,:]
        svhn_test_data = svhn_test_data[:,2:30,2:30,:]
        svhn_valid_data = svhn_valid_data[:,2:30,2:30,:]
        
        

        datasets['svhn']['train'] = {'images': svhn_train_data, 'labels': svhn_train_label}
        datasets['svhn']['test'] = {'images': svhn_test_data, 'labels': svhn_test_label}
        datasets['svhn']['valid'] = {'images': svhn_valid_data, 'labels': svhn_valid_label}
            
    if sets.has_key('mnistm'):
        if 'mnist' not in locals():
            mnist = input_data.read_data_sets(data_dir + 'MNIST_data', one_hot=True)  
        mnistm = pickle.load(open(data_dir + 'MNISTM/mnistm_data.pkl', 'rb'))
        mnistm_train = mnistm['train']
        mnistm_test = mnistm['test']
        mnistm_valid = mnistm['valid']
        
        datasets['mnistm']['train'] = {'images': mnistm_train, 'labels': mnist.train.labels}
        datasets['mnistm']['test'] = {'images': mnistm_test, 'labels': mnist.test.labels}
        datasets['mnistm']['valid'] = {'images': mnistm_valid, 'labels': mnist.validation.labels}
    if sets.has_key('usps'):
        usps_file =  open(data_dir + 'USPS/usps_28x28.pkl', 'rb')
        usps = pickle.load(usps_file)
        n = 5104
        usps_train = (usps[0][0][:n].reshape(-1,28,28,1)*255.).astype('uint8')
        usps_train = np.concatenate([usps_train, usps_train, usps_train], 3)
        usps_valid = (usps[0][0][n:].reshape(-1,28,28,1)*255.).astype('uint8')
        usps_valid = np.concatenate([usps_valid, usps_valid, usps_valid], 3)
        usps_test = (usps[1][0].reshape(-1,28,28,1)*255.).astype('uint8')
        usps_test = np.concatenate([usps_test, usps_test, usps_test], 3)
        usps_images = (np.concatenate([usps[0][0], usps[1][0]]).reshape(-1, 28, 28, 1) * 255.).astype(np.uint8)
        usps_images = np.concatenate([usps_images, usps_images, usps_images], 3)
        
        datasets['usps']['train'] = {'images': usps_train, 'labels': to_one_hot(usps[0][1][:n])}
        datasets['usps']['test'] = {'images': usps_test, 'labels': to_one_hot(usps[1][1])}
        datasets['usps']['valid'] = {'images': usps_valid, 'labels': to_one_hot(usps[0][1][n:])}

    if sets.has_key('cifar'):
        batch_1 = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/cifar-10-batches-mat/data_batch_1.mat')
        batch_2 = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/cifar-10-batches-mat/data_batch_2.mat')
        batch_3 = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/cifar-10-batches-mat/data_batch_3.mat')
        batch_4 = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/cifar-10-batches-mat/data_batch_4.mat')
        batch_5 = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/cifar-10-batches-mat/data_batch_5.mat')
        batch_test = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/cifar-10-batches-mat/test_batch.mat')

        train_batch = np.concatenate([batch_1['data'], batch_2['data'], batch_3['data'], 
                                     batch_4['data'], batch_5['data']])
        train_label = np.concatenate([batch_1['labels'], batch_2['labels'], batch_3['labels'], 
                                     batch_4['labels'], batch_5['labels']])

        cifar_train_data = np.reshape(train_batch, [-1, 3, 32, 32]).transpose((0,2,3,1))
        cifar_train_label = to_one_hot(np.squeeze(train_label))

        cifar_train_data_reduce = cifar_train_data[cifar_train_label[:,6]==0]
        cifar_train_label_tmp = cifar_train_label[cifar_train_label[:,6]==0]
        cifar_train_label_reduce = np.concatenate([cifar_train_label_tmp[:,:6], cifar_train_label_tmp[:,7:]], axis=1)

        # cifar_valid_data = cifar_train_data[-5000:]
        # cifar_train_data = cifar_train_data[:-5000]
        
        # cifar_valid_label = cifar_train_label[-5000:]
        # cifar_train_label = cifar_train_label[:-5000]

        cifar_valid_data_reduce = cifar_train_data_reduce[-5000:]
        cifar_train_data_reduce = cifar_train_data_reduce[:-5000]
        
        cifar_valid_label_reduce = cifar_train_label_reduce[-5000:]
        cifar_train_label_reduce = cifar_train_label_reduce[:-5000]

        cifar_test_data = np.reshape(batch_test['data'], [-1, 3, 32, 32]).transpose((0,2,3,1))
        cifar_test_label = to_one_hot(np.squeeze(batch_test['labels']))

        cifar_test_data_reduce = cifar_test_data[cifar_test_label[:,6]==0]
        cifar_test_label_tmp = cifar_test_label[cifar_test_label[:,6]==0]
        cifar_test_label_reduce = np.concatenate([cifar_test_label_tmp[:,:6], cifar_test_label_tmp[:,7:]], axis=1)

        datasets['cifar']['train'] = {'images': cifar_train_data_reduce, 'labels': cifar_train_label_reduce}
        datasets['cifar']['test'] = {'images': cifar_test_data_reduce, 'labels': cifar_test_label_reduce}
        datasets['cifar']['valid'] = {'images': cifar_valid_data_reduce, 'labels': cifar_valid_label_reduce}

    if sets.has_key('stl'):
        stl_train = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/stl10_matlab/train.mat')
        stl_train_data = np.reshape(stl_train['X'], [-1, 3, 96, 96]).transpose((0,3,2,1))

        stl_train_label = np.squeeze(stl_train['y']-1)

        stl_train_label_tmp = np.zeros([stl_train_label.shape[0], 10])

        stl_train_label_tmp[stl_train_label==0,0]=1.
        stl_train_label_tmp[stl_train_label==1,2]=1.
        stl_train_label_tmp[stl_train_label==2,1]=1.
        stl_train_label_tmp[stl_train_label==3,3]=1.
        stl_train_label_tmp[stl_train_label==4,4]=1.
        stl_train_label_tmp[stl_train_label==5,5]=1.
        stl_train_label_tmp[stl_train_label==6,7]=1.
        stl_train_label_tmp[stl_train_label==7,6]=1.
        stl_train_label_tmp[stl_train_label==8,8]=1.
        stl_train_label_tmp[stl_train_label==9,9]=1.


        stl_train_data_reduce = stl_train_data[stl_train_label_tmp[:,6]==0]
        stl_train_label_tmp = stl_train_label_tmp[stl_train_label_tmp[:,6]==0]
        stl_train_label_reduce = np.concatenate([stl_train_label_tmp[:,:6], stl_train_label_tmp[:,7:]], axis=1)


        stl_test = scipy.io.loadmat('/home/yl353/Peter/new_domain/data/stl10_matlab/test.mat')
        stl_test_data = np.reshape(stl_test['X'], [-1, 3, 96, 96]).transpose((0,3,2,1))
        
        stl_test_label = np.squeeze(stl_test['y']-1)

        stl_test_label_tmp = np.zeros([stl_test_label.shape[0], 10])

        stl_test_label_tmp[stl_test_label==0,0]=1.
        stl_test_label_tmp[stl_test_label==1,2]=1.
        stl_test_label_tmp[stl_test_label==2,1]=1.
        stl_test_label_tmp[stl_test_label==3,3]=1.
        stl_test_label_tmp[stl_test_label==4,4]=1.
        stl_test_label_tmp[stl_test_label==5,5]=1.
        stl_test_label_tmp[stl_test_label==6,7]=1.
        stl_test_label_tmp[stl_test_label==7,6]=1.
        stl_test_label_tmp[stl_test_label==8,8]=1.
        stl_test_label_tmp[stl_test_label==9,9]=1.

        stl_test_data_reduce = stl_test_data[stl_test_label_tmp[:,6]==0]
        stl_test_label_tmp = stl_test_label_tmp[stl_test_label_tmp[:,6]==0]
        stl_test_label_reduce = np.concatenate([stl_test_label_tmp[:,:6], stl_test_label_tmp[:,7:]], axis=1)
        

        stl_valid_data_reduce = stl_train_data_reduce[-500:]
        stl_train_data_reduce = stl_train_data_reduce[:-500]
        
        stl_valid_label_reduce = stl_train_label_reduce[-500:]
        stl_train_label_reduce = stl_train_label_reduce[:-500]

        stl_train_data_reduce = [np.expand_dims(cv2.resize(x, dsize=(32,32)), 0) for x in stl_train_data_reduce]
        stl_train_data_reduce = np.concatenate(stl_train_data_reduce)
        stl_test_data_reduce = [np.expand_dims(cv2.resize(x, dsize=(32,32)), 0) for x in stl_test_data_reduce]
        stl_test_data_reduce = np.concatenate(stl_test_data_reduce)
        stl_valid_data_reduce = [np.expand_dims(cv2.resize(x, dsize=(32,32)), 0) for x in stl_valid_data_reduce]
        stl_valid_data_reduce = np.concatenate(stl_valid_data_reduce)

        datasets['stl']['train'] = {'images': stl_train_data_reduce, 'labels': stl_train_label_reduce}
        datasets['stl']['test'] = {'images': stl_test_data_reduce, 'labels': stl_test_label_reduce}
        datasets['stl']['valid'] = {'images': stl_valid_data_reduce, 'labels': stl_valid_label_reduce}

    return datasets


def save_dataset(datasets,save_path = './save_datasets/'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for key in datasets.keys():
        train = datasets[key]['train']['images']
        valid = datasets[key]['valid']['images']
        test = datasets[key]['test']['images']
        labtrain = datasets[key]['train']['labels']
        labval = datasets[key]['valid']['labels']
        labtest = datasets[key]['test']['labels']
        scipy.io.savemat(save_path + key + '.mat',{'train':train, 'val':valid,'test':test,'labtrain':labtrain,'labval':labval,'labtest':labtest})
    return 0

def sets_concatenate(datasets, sets):
    N_train = 0
    N_valid = 0
    N_test = 0
    
    for key in sets:
        label_len = datasets[key]['train']['labels'].shape[1]
        N_train = N_train + datasets[key]['train']['images'].shape[0]
        N_valid = N_valid + datasets[key]['valid']['images'].shape[0]
        N_test = N_test + datasets[key]['test']['images'].shape[0]
        S = datasets[key]['train']['images'].shape[1]
    train = {'images': np.zeros((N_train,S,S,3)).astype(np.float32),'labels':np.zeros((N_train,label_len)).astype('float32'),'domains':np.zeros((N_train,)).astype('float32')}
    valid = {'images': np.zeros((N_valid,S,S,3)).astype(np.float32),'labels':np.zeros((N_valid,label_len)).astype('float32'),'domains':np.zeros((N_valid,)).astype('float32')}
    test = {'images': np.zeros((N_test,S,S,3)).astype(np.float32),'labels':np.zeros((N_test,label_len)).astype('float32'),'domains':np.zeros((N_test,)).astype('float32')}
    srt = 0
    edn = 0
    for key in sets:
        domain = sets[key]
        srt = edn
        edn = srt + datasets[key]['train']['images'].shape[0]
        train['images'][srt:edn,:,:,:] = datasets[key]['train']['images']
        train['labels'][srt:edn,:] = datasets[key]['train']['labels']
        train['domains'][srt:edn] = domain * np.ones((datasets[key]['train']['images'].shape[0],)).astype('float32')
    srt = 0
    edn = 0
    for key in sets:
        domain = sets[key]
        srt = edn
        edn = srt + datasets[key]['valid']['images'].shape[0]
        valid['images'][srt:edn,:,:,:] = datasets[key]['valid']['images']
        valid['labels'][srt:edn,:] = datasets[key]['valid']['labels']
        valid['domains'][srt:edn] = domain * np.ones((datasets[key]['valid']['images'].shape[0],)).astype('float32')
    srt = 0
    edn = 0
    for key in sets:
        domain = sets[key]
        srt = edn
        edn = srt + datasets[key]['test']['images'].shape[0]
        test['images'][srt:edn,:,:,:] = datasets[key]['test']['images']
        test['labels'][srt:edn,:] = datasets[key]['test']['labels']
        test['domains'][srt:edn] = domain * np.ones((datasets[key]['test']['images'].shape[0],)).astype('float32')
    return train, valid, test

def source_target(datasets, sources, targets, unify_source = False):
    N1 = len(sources.keys())
    N_domain = N1 + len(targets.keys())
    domain_idx = 0
    for key in sources.keys():
        sources[key] = domain_idx
        domain_idx = domain_idx + 1
    for key in targets.keys():
        targets[key] = domain_idx    
        domain_idx = domain_idx + 1
        
    source_train, source_valid, source_test = sets_concatenate(datasets, sources)
    target_train, target_valid, target_test = sets_concatenate(datasets, targets)
    
    if unify_source:
        source_train['domains'] = to_one_hot(0 * source_train['domains'], 2)
        source_valid['domains'] = to_one_hot(0 * source_valid['domains'], 2)
        source_test['domains'] = to_one_hot(0 * source_test['domains'], 2)
        target_train['domains'] = to_one_hot(0 * target_train['domains'] + 1, 2)
        target_valid['domains'] = to_one_hot(0 * target_valid['domains'] + 1, 2)
        target_test['domains'] = to_one_hot(0 * target_test['domains'] + 1, 2)
    else:
        source_train['domains'] = to_one_hot(source_train['domains'], N_domain)
        source_valid['domains'] = to_one_hot(source_valid['domains'], N_domain)
        source_test['domains'] = to_one_hot(source_test['domains'], N_domain)
        target_train['domains'] = to_one_hot(target_train['domains'], N_domain)
        target_valid['domains'] = to_one_hot(target_valid['domains'], N_domain)
        target_test['domains'] = to_one_hot(target_test['domains'], N_domain)
    return source_train, source_valid, source_test, target_train, target_valid, target_test

def normalize(data):
    image_mean = data - np.expand_dims(np.expand_dims(data.mean((1,2)),1),1)
    image_std = np.sqrt((image_mean**2).mean((1,2))+1e-8)
    return image_mean / np.expand_dims(np.expand_dims(image_std,1),1)

def normalize_dataset(datasets, t = 'norm'):
    if t is 'mean':
        temp_data = []
        for key in datasets.keys():
            temp_data.append(datasets[key]['train']['images'])
        temp_data = np.concatenate(temp_data)
        image_mean = temp_data.mean((0, 1, 2))
        image_mean = image_mean.astype('float32')
        for key in datasets.keys():
            datasets[key]['train']['images'] = (datasets[key]['train']['images'].astype('float32') - image_mean)/255.
            datasets[key]['valid']['images'] = (datasets[key]['valid']['images'].astype('float32')  - image_mean)/255.
            datasets[key]['test']['images'] = (datasets[key]['test']['images'].astype('float32')  - image_mean)/255.
    elif t is 'standard':
        for key in datasets.keys():
            datasets[key]['train']['images'] = (datasets[key]['train']['images'].astype('float32'))/255.
            datasets[key]['valid']['images'] = (datasets[key]['valid']['images'].astype('float32'))/255.
            datasets[key]['test']['images'] = (datasets[key]['test']['images'].astype('float32'))/255.
    elif t is 'none':
        datasets = datasets
    elif t is 'individual':
        for key in datasets.keys():
            temp_data = datasets[key]['train']['images']
            image_mean = temp_data.mean((0, 1, 2))
            image_mean = image_mean.astype('float32')
            datasets[key]['train']['images'] = (datasets[key]['train']['images'].astype('float32') - image_mean)/255.
            datasets[key]['valid']['images'] = (datasets[key]['valid']['images'].astype('float32')  - image_mean)/255.
            datasets[key]['test']['images'] = (datasets[key]['test']['images'].astype('float32')  - image_mean)/255.
    elif t is 'norm':
        for key in datasets.keys():
            if key =='mnist':
                tmp_1 = datasets[key]['train']['images'][:(len(datasets[key]['train']['images']) // 2)]
                tmp_2 = datasets[key]['train']['images'][(len(datasets[key]['train']['images']) // 2):]
                datasets[key]['train']['images'] = np.concatenate([normalize(tmp_1),normalize(tmp_2)])
            else:
                datasets[key]['train']['images'] = normalize(datasets[key]['train']['images'])
            
            datasets[key]['valid']['images'] = normalize(datasets[key]['valid']['images'])
            datasets[key]['test']['images'] = normalize(datasets[key]['test']['images'])

    return datasets

def source_target_separate(datasets, sources, targets):
    N1 = len(sources.keys())
    N_domain = N1 + len(targets.keys())
    domain_idx = 0
    sets = {}
    for key in sources.keys():
        sources[key] = domain_idx
        sets[key] = domain_idx
        domain_idx = domain_idx + 1
    for key in targets.keys():
        targets[key] = domain_idx    
        sets[key] = domain_idx
        domain_idx = domain_idx + 1
    for key in datasets.keys():
        datasets[key]['train']['domains'] = to_one_hot(sets[key] * np.ones((datasets[key]['train']['images'].shape[0],)).astype('float32'), N_domain)
        datasets[key]['valid']['domains'] = to_one_hot(sets[key] * np.ones((datasets[key]['valid']['images'].shape[0],)).astype('float32'), N_domain)
        datasets[key]['test']['domains'] = to_one_hot(sets[key] * np.ones((datasets[key]['test']['images'].shape[0],)).astype('float32'), N_domain)   
    return datasets     

def source_target_separate_baseline(datasets, sources, targets):
    N1 = len(sources.keys())
    N_domain = N1 + len(targets.keys())
    domain_idx = 0
    domains = {}
    for key in sources.keys():
        sources[key] = domain_idx
        domains[key] = domain_idx
        domain_idx = domain_idx + 1
    for key in targets.keys():
        targets[key] = domain_idx  
        domains[key] = domain_idx  
    for key in datasets.keys():
        datasets[key]['train']['domains'] = to_one_hot(domains[key] * np.ones((datasets[key]['train']['images'].shape[0],)).astype('float32'), 2)
        datasets[key]['valid']['domains'] = to_one_hot(domains[key] * np.ones((datasets[key]['valid']['images'].shape[0],)).astype('float32'), 2)
        datasets[key]['test']['domains'] = to_one_hot(domains[key] * np.ones((datasets[key]['test']['images'].shape[0],)).astype('float32'), 2) 
    return datasets  






