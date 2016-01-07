
# coding: utf-8

# In[ ]:

import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np

import os


# In[ ]:

dev = mx.gpu(0)
batch_size = 128
num_gpus = 1

train_iter = mx.io.ImageRecordIter(
    shuffle=True,
    path_imgrec="./data/cifar/train.rec",
    mean_r = 128,
    mean_g = 128,
    mean_b = 128,
    scale = 0.0078125,
    rand_crop=True,
    rand_mirror=True,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2)

val_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/cifar/test.rec",
    mean_r = 128,
    mean_g = 128,
    mean_b = 128,
    scale = 0.0078125,
    rand_crop=False,
    rand_mirror=False,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2,
    round_batch=False)


# In[ ]:

def Softmax(arr):
    max_val = np.max(arr, axis=1, keepdims=True)
    tmp = arr - max_val
    exp = np.exp(tmp)
    norm = np.sum(exp, axis=1, keepdims=True)
    return exp / norm


# In[ ]:

def SoftmaxGrad(arr, idx):
    grad = np.copy(arr)
    for i in range(arr.shape[0]):
        p = grad[i, idx]
        grad[i, :] *= -p
        grad[i, idx] = p * (1. - p)
    return grad


# In[ ]:

def SGD(weight, grad, lr=0.1, wd=0.0001, grad_norm=batch_size):
    weight[:] -= lr * (mx.nd.clip(grad, -5, 5) / batch_size + wd * weight)


# In[ ]:

def LogLossGrad(arr, label):
    grad = np.copy(arr)
    for i in range(arr.shape[0]):
        grad[i, label[i]] -= 1.
    return grad


# In[ ]:

def CalAcc(pred_prob, label):
    pred = np.argmax(pred_prob, axis=1)
    return np.sum(pred == label) * 1.0


# In[ ]:

def CalLoss(pred_prob, label):
    loss = 0.
    for i in range(pred_prob.shape[0]):
        loss += -np.log(max(pred_prob[i, label[i]], 1e-10))
    return loss


# In[ ]:

def ConvFactory(data, kernel, pad, num_filter, stride=1):
    conv = mx.sym.Convolution(data=data, kernel=(kernel, kernel), pad=(pad, pad), stride=(stride, stride), num_filter=num_filter)
    bn = mx.sym.BatchNorm(data=conv)
    act = mx.sym.Activation(data=bn, act_type='relu')
    return act

def ConvFactoryOld(data, kernel, pad, num_filter, stride=1):
    conv = mx.sym.Convolution(data=data, kernel=(kernel, kernel), pad=(pad, pad), stride=(stride, stride), num_filter=num_filter)
    act = mx.sym.Activation(data=conv, act_type='relu')
    return act



# In[ ]:

def acc_normal_old(model, val_iter, arg_map, grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_samp = 0
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        arg_map["data"][:] = data

        model.forward(is_train=False)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        val_acc += CalAcc(alpha, label.asnumpy())
        num_samp += batch_size
    return(val_acc / num_samp)


# In[ ]:

def acc_normal(net, fea, val_iter, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_samp = 0
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        net_arg_map["data"][:] = data
        net.forward(is_train=False)

        fea_arg_map["feature"][:] = net.outputs[0]
        fea.forward(is_train=False)

        theta = fea.outputs[0].asnumpy()
        alpha = Softmax(theta)
        val_acc += CalAcc(alpha, label.asnumpy())
        num_samp += batch_size
    return(val_acc / num_samp)


# In[ ]:

def acc_perb_L0(net, fea, val_iter, coe_pb, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_samp = 0
    nn=0
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        net_arg_map["data"][:] = data
        net.forward(is_train=True)

        fea_arg_map["feature"][:] = net.outputs[0]
        fea.forward(is_train=True)
        theta = fea.outputs[0].asnumpy()
        alpha = Softmax(theta)

        grad = LogLossGrad(alpha, label.asnumpy())
        fea_out_grad[:] = grad
        fea.backward([fea_out_grad])

        net.backward([fea_grad_map["feature"]])

        noise = np.sign(net_grad_map["data"].asnumpy())

        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])): #1： #
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0

        pdata = data.asnumpy() + coe_pb * noise
        net_arg_map["data"][:] = pdata
        net.forward(is_train=False)
        fea_arg_map["feature"][:] = net.outputs[0]
        fea.forward(is_train=False)
        raw_output = fea.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        val_acc += CalAcc(pred, label.asnumpy())
        num_samp += batch_size
    if  nn>0:
        print('L0 gradien being 0 :', nn)
    return(val_acc / num_samp)


# In[ ]:

def acc_perb_L2(net, fea, val_iter, coe_pb, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_batch = 0
    nn=0
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        net_arg_map["data"][:] = data

        net.forward(is_train=True)
        fea_arg_map["feature"][:] = net.outputs[0]
        fea.forward(is_train=True)
        theta = fea.outputs[0].asnumpy()
        alpha = Softmax(theta)

        grad = LogLossGrad(alpha, label.asnumpy())
        fea_out_grad[:] = grad
        fea.backward([fea_out_grad])
        net.backward([fea_grad_map["feature"]])

        noise = net_grad_map["data"].asnumpy()

        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) == 0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])): #1： #
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0
        pdata = data.asnumpy() + coe_pb * noise
        net_arg_map["data"][:] = pdata
        net.forward(is_train=False)
        fea_arg_map["feature"][:] = net.outputs[0]
        fea.forward(is_train=False)
        raw_output = fea.outputs[0].asnumpy()
        pred = Softmax(raw_output)

        val_acc += CalAcc(pred, label.asnumpy()) /  batch_size
        num_batch += 1
    if  nn>0:
        print('L2 gradien being 0 :', nn)
    return(val_acc / num_batch)


# In[ ]:

def acc_perb_alpha(net, fea, val_iter, coe_pb, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_samp = 0
    nn=0
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]

        T = np.zeros((10, batch_size, data_shape[1], data_shape[2], data_shape[3]))
        noise = np.zeros(data.shape)
        #===================
        for i in range(10):
            net_arg_map["data"][:] = data
            net.forward(is_train=True)
            fea_arg_map["feature"][:] = net.outputs[0]
            fea.forward(is_train=True)
            theta = fea.outputs[0].asnumpy()
            alpha = Softmax(theta)

            grad = LogLossGrad(alpha, i*np.ones(alpha.shape[0]))
            for j in range(batch_size):
                grad[j] = -alpha[j][i]*grad[j]
            fea_out_grad[:] = grad
            fea.backward([fea_out_grad])
            net.backward([fea_grad_map["feature"]])
            #print(data_grad.asnumpy().shape)
            T[i] = net_grad_map["data"].asnumpy()

        for j in range(batch_size):
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])): #1： #
                perb_scale = np.zeros(10)
                for i in range(10):
                    if (i == y):
                        perb_scale[i] = np.inf
                    else:
                        perb_scale[i] = (alpha[j][y] - alpha[j][i])/np.linalg.norm((T[i][j]-T[y][j]).flatten(),2)
                noise[j] = T[np.argmin(perb_scale)][j]-T[y][j]
        #====================
        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            else:
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
        pdata = data.asnumpy() + coe_pb * noise
        net_arg_map["data"][:] = pdata
        net.forward(is_train=False)
        fea_arg_map["feature"][:] = net.outputs[0]
        fea.forward(is_train=False)
        raw_output = fea.outputs[0].asnumpy()
        pred = Softmax(raw_output)

        val_acc += CalAcc(pred, label.asnumpy()) /batch_size
        num_samp += 1
    if  nn>0:
        print('Alpha gradien being 0 :', nn)
    return(val_acc / num_samp)


# ## Generate Fixed Perturbed Data

# In[ ]:

data = mx.sym.Variable('data')
conv1 = ConvFactoryOld(data, 3, 1, 64)
conv2 = ConvFactoryOld(conv1, 3, 1, 64)
conv3 = ConvFactoryOld(conv2, 3, 1, 64)
mp1 = mx.sym.Pooling(data=conv3, pool_type="max", kernel=(3,3), stride=(2,2))
conv4 = ConvFactoryOld(mp1, 3, 1, 128)
conv5 = ConvFactoryOld(conv4, 3, 1, 128)
conv6 = ConvFactoryOld(conv5, 3, 1, 128)
mp1 = mx.sym.Pooling(data=conv6, pool_type="max", kernel=(3,3), stride=(2,2))
fl = mx.sym.Flatten(data=mp1)
fc1 = mx.sym.FullyConnected(data=fl, num_hidden=2048)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

fc2 = mx.sym.FullyConnected(data=act1, num_hidden=2048)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

flatten = mx.sym.FullyConnected(data=act2, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 3, 28, 28)
arg_names = flatten.list_arguments()
arg_shapes, output_shapes, aux_shapes = flatten.infer_shape(data=data_shape)
arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_sum = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
aux_states =  [mx.nd.zeros(shape, ctx=dev) for shape in aux_shapes]
pred = mx.nd.zeros(output_shapes[0])
reqs = ["write" for name in arg_names]

model = flatten.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs, aux_states=aux_states)
arg_map = dict(zip(arg_names, arg_arrays))
grad_map = dict(zip(arg_names, grad_arrays))
sum_map = dict(zip(arg_names, grad_sum))
data_grad = grad_map["data"]
out_grad = mx.nd.zeros(model.outputs[0].shape, ctx=dev)


# In[ ]:

for name in arg_names:
    if "weight" in name:
        arr = arg_map[name]
        shape = arr.shape
        fan_in, fan_out = np.prod(shape[1:]), shape[0]
        factor = fan_in
        scale = np.sqrt(3. / factor)
        arr[:] = mx.rnd.uniform(-scale, scale, arr.shape)
    elif "gamma" in name:
        arr = arg_map[name]
        arr[:] = 1.0
    else:
        arr = arg_map[name]
        arr[:] = 0.

# In[ ]:
num_round = 30
train_acc = 0.
nbatch = 0
lr = 0.03
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]

        arg_map["data"][:] = data
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        train_acc += CalAcc(alpha, label.asnumpy()) / batch_size
        train_loss += CalLoss(alpha, label.asnumpy()) / batch_size
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
        # data_grad[:] = grad_map["data"]
        for name in arg_names:
            if name != "data":
                if name.endswith("weight"):
                    SGD(arg_map[name], grad_map[name], lr)
                else:
                    SGD(arg_map[name], grad_map[name], lr, 0)

        nbatch += 1
    train_acc /= nbatch
    train_loss /= nbatch
    val_acc = acc_normal_old(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

num_round = 20
train_acc = 0.
nbatch = 0
lr = 0.003
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]

        arg_map["data"][:] = data
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        train_acc += CalAcc(alpha, label.asnumpy()) / batch_size
        train_loss += CalLoss(alpha, label.asnumpy()) / batch_size
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
        # data_grad[:] = grad_map["data"]
        for name in arg_names:
            if name != "data":
                if name.endswith("weight"):
                    SGD(arg_map[name], grad_map[name], lr)
                else:
                    SGD(arg_map[name], grad_map[name], lr, 0)

        nbatch += 1
    #print(np.linalg.norm(data_grad.asnumpy(), 2))
    train_acc /= nbatch
    train_loss /= nbatch
    val_acc = acc_normal_old(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

val_iter.reset()
val_acc = 0.0
val_acc_pb = 0.0
coe_pb = 0.5
num_samp = 0

perb_data = []
perb_lab = []

for dbatch in val_iter:
    data = dbatch.data[0]
    label = dbatch.label[0]
    arg_map["data"][:] = data
    batch_size = label.asnumpy().shape[0]

    model.forward(is_train=False)
    theta = model.outputs[0].asnumpy()
    alpha = Softmax(theta)
    val_acc += CalAcc(alpha, label.asnumpy())
    #########
    grad = LogLossGrad(alpha, label.asnumpy())
    out_grad[:] = grad
    model.backward([out_grad])
    noise = data_grad.asnumpy()
    for j in range(batch_size):
        noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
    pdata = data.asnumpy() + coe_pb * noise
    arg_map["data"][:] = pdata
    model.forward(is_train=True)
    raw_output = model.outputs[0].asnumpy()
    pred = Softmax(raw_output)
    val_acc_pb += CalAcc(pred, label.asnumpy())
    num_samp += batch_size

    perb_data.append(pdata)
    perb_lab.append(label.asnumpy())
print("Val Batch Accuracy: ", val_acc / num_samp)
print("Val Batch Accuracy after pertubation: ", val_acc_pb / num_samp)
print(acc_normal_old(model, val_iter,arg_map, grad_map))

pdata = np.concatenate(perb_data, axis = 0)
plabel = np.concatenate(perb_lab, axis = 0)
perb_iter = mx.io.NDArrayIter(
    data = pdata,
    label = plabel,
    batch_size = 128,
    shuffle = False)


# # Smaller VGG-D Network

# In[ ]:

net = mx.sym.Variable('data')
net = ConvFactory(net, 3, 1, 64)
net = ConvFactory(net, 3, 1, 64)
net = ConvFactory(net, 3, 1, 64)
net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3,3), stride=(2,2))
net = ConvFactory(net, 3, 1, 128)
net = ConvFactory(net, 3, 1, 128)
net = ConvFactory(net, 3, 1, 128)
net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3,3), stride=(2,2))
net = mx.sym.Flatten(data=net)
net = mx.sym.FullyConnected(data=net, num_hidden=2048)
net = mx.sym.Activation(data=net, act_type="relu")
net = mx.sym.Dropout(data=net)

fea = mx.sym.Variable('feature')
fea = mx.sym.FullyConnected(data=fea, num_hidden=2048)
fea = mx.sym.Activation(data=fea, act_type="relu")
fea = mx.sym.Dropout(data=fea)
fea = mx.sym.FullyConnected(data=fea, num_hidden=10)


# In[ ]:

net_data_shape = (batch_size, 3, 28, 28)
net_arg_names = net.list_arguments()
net_arg_shapes, net_output_shapes, net_aux_shapes = net.infer_shape(data=net_data_shape)

net_arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in net_arg_shapes]
net_grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in net_arg_shapes]
net_aux_states =  [mx.nd.zeros(shape, ctx=dev) for shape in net_aux_shapes]

reqs = ["write" for name in net_arg_names]

net_model = net.bind(ctx=dev, args=net_arg_arrays, args_grad=net_grad_arrays, grad_req=reqs, aux_states=net_aux_states)
net_arg_map = dict(zip(net_arg_names, net_arg_arrays))
net_grad_map = dict(zip(net_arg_names, net_grad_arrays))
net_data_grad = net_grad_map["data"]
net_out_grad = mx.nd.zeros(net_model.outputs[0].shape, ctx=dev)


# In[ ]:

fea_shape = (batch_size, net_output_shapes[0][1])
fea_arg_names = fea.list_arguments()
fea_arg_shapes, fea_output_shapes, fea_aux_shapes = fea.infer_shape(feature=fea_shape)
fea_arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in fea_arg_shapes]
fea_grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in fea_arg_shapes]
fea_aux_states =  [mx.nd.zeros(shape, ctx=dev) for shape in fea_aux_shapes]

reqs = ["write" for name in fea_arg_names]

fea_model = fea.bind(ctx=dev, args=fea_arg_arrays, args_grad=fea_grad_arrays, grad_req=reqs, aux_states=fea_aux_states)
fea_arg_map = dict(zip(fea_arg_names, fea_arg_arrays))
fea_grad_map = dict(zip(fea_arg_names, fea_grad_arrays))
fea_grad = fea_grad_map["feature"]
fea_out_grad = mx.nd.zeros(fea_model.outputs[0].shape, ctx=dev)


# In[ ]:

for name in net_arg_names:
    if "weight" in name:
        arr = net_arg_map[name]
        shape = arr.shape
        fan_in, fan_out = np.prod(shape[1:]), shape[0]
        factor = fan_in
        scale = np.sqrt(6. / factor)
        arr[:] = mx.rnd.uniform(-scale, scale, arr.shape)
    elif "gamma" in name:
        arr = net_arg_map[name]
        arr[:] = 1.0
    else:
        arr = net_arg_map[name]
        arr[:] = 0.

for name in fea_arg_names:
    if "weight" in name:
        arr = fea_arg_map[name]
        shape = arr.shape
        fan_in, fan_out = np.prod(shape[1:]), shape[0]
        factor = fan_in
        scale = np.sqrt(6. / factor)
        arr[:] = mx.rnd.uniform(-scale, scale, arr.shape)
    elif "gamma" in name:
        arr = fea_arg_map[name]
        arr[:] = 1.0
    else:
        arr = fea_arg_map[name]
        arr[:] = 0.


# In[ ]:

num_round = 100
train_acc = 0.
nbatch = 0
lr=0.045
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        net_arg_map["data"][:] = data
        net_model.forward(is_train=True)
        feature = net_model.outputs[0].asnumpy()

        fea_arg_map["feature"][:] = feature
        fea_model.forward(is_train=True)
        raw_output = fea_model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        train_acc += CalAcc(pred, label.asnumpy()) / batch_size
        train_loss += CalLoss(pred, label.asnumpy()) / batch_size

        grad = LogLossGrad(pred, label.asnumpy())

        fea_out_grad[:] = grad
        fea_model.backward([fea_out_grad])
        noise = fea_grad.asnumpy()

        for j in range(batch_size):
            if np.sum(np.abs(noise[j,:])) < 1e-4:
                noise[j,:] = 0
            else:
                noise[j,:] = (0.25 * np.linalg.norm(feature[j,:].flatten(),2) / np.linalg.norm(noise[j, :].flatten(), 2)) * noise[j, :]
        fea_arg_map["feature"][:] = feature + noise

        fea_model.forward(is_train=True)
        raw_output = fea_model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        grad = LogLossGrad(pred, label.asnumpy())
        fea_out_grad[:] = grad
        fea_model.backward([fea_out_grad])

        net_model.backward([fea_grad])
        for name in net_arg_names:
            if name != "data":
                if name.endswith("weight"):
                    SGD(net_arg_map[name], net_grad_map[name], lr)
                else:
                    SGD(net_arg_map[name], net_grad_map[name], lr, 0)
        for name in fea_arg_names:
            if name != "feature":
                if name.endswith("weight"):
                    SGD(fea_arg_map[name], fea_grad_map[name], lr)
                else:
                    SGD(fea_arg_map[name], fea_grad_map[name], lr, 0)
        nbatch += 1

    train_acc /= nbatch
    train_loss /= nbatch
    print("[%d] Train Accuracy: %.4f\t Train Loss: %.5f" % (i, train_acc, train_loss))


# In[ ]:
print('-' * 60)
print('Normal Validation: %.3f' % acc_normal(net_model, fea_model, val_iter, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('Fixed set perturbation: %.3f' % acc_normal(net_model, fea_model, perb_iter, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('L0 perturbation: %.3f' % acc_perb_L0(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('L2 perturbation: %.3f' % acc_perb_L2(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('Alpha perturbation: %.3f' % acc_perb_alpha(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))


# In[ ]:

num_round = 40
train_acc = 0.
nbatch = 0
lr=0.01
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        net_arg_map["data"][:] = data
        net_model.forward(is_train=True)
        feature = net_model.outputs[0].asnumpy()

        fea_arg_map["feature"][:] = feature
        fea_model.forward(is_train=True)
        raw_output = fea_model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        train_acc += CalAcc(pred, label.asnumpy()) / batch_size
        train_loss += CalLoss(pred, label.asnumpy()) / batch_size

        grad = LogLossGrad(pred, label.asnumpy())

        fea_out_grad[:] = grad
        fea_model.backward([fea_out_grad])
        noise = fea_grad.asnumpy()

        for j in range(batch_size):
            if np.sum(np.abs(noise[j,:])) < 1e-4:
                noise[j,:] = 0
            else:
                noise[j,:] = (0.25 *np.linalg.norm(feature[j,:].flatten(),2) / np.linalg.norm(noise[j, :].flatten(), 2)) * noise[j, :]
        fea_arg_map["feature"][:] = feature + noise

        fea_model.forward(is_train=True)
        raw_output = fea_model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        grad = LogLossGrad(pred, label.asnumpy())
        fea_out_grad[:] = grad
        fea_model.backward([fea_out_grad])

        net_model.backward([fea_grad])
        for name in net_arg_names:
            if name != "data":
                if name.endswith("weight"):
                    SGD(net_arg_map[name], net_grad_map[name], lr)
                else:
                    SGD(net_arg_map[name], net_grad_map[name], lr, 0)
        for name in fea_arg_names:
            if name != "feature":
                if name.endswith("weight"):
                    SGD(fea_arg_map[name], fea_grad_map[name], lr)
                else:
                    SGD(fea_arg_map[name], fea_grad_map[name], lr, 0)
        nbatch += 1

    train_acc /= nbatch
    train_loss /= nbatch
    print("[%d] Train Accuracy: %.4f\t Train Loss: %.5f" % (i, train_acc, train_loss))


# In[ ]:
print('-' * 60)
print('Normal Validation: %.3f' % acc_normal(net_model, fea_model, val_iter, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('Fixed set perturbation: %.3f' % acc_normal(net_model, fea_model, perb_iter, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('L0 perturbation: %.3f' % acc_perb_L0(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('L2 perturbation: %.3f' % acc_perb_L2(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('Alpha perturbation: %.3f' % acc_perb_alpha(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))

"""
# In[ ]:

num_round = 20
train_acc = 0.
nbatch = 0
lr=0.001
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        net_arg_map["data"][:] = data
        net_model.forward(is_train=True)
        feature = net_model.outputs[0].asnumpy()

        fea_arg_map["feature"][:] = feature
        fea_model.forward(is_train=True)
        raw_output = fea_model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        train_acc += CalAcc(pred, label.asnumpy()) / batch_size
        train_loss += CalLoss(pred, label.asnumpy()) / batch_size

        grad = LogLossGrad(pred, label.asnumpy())

        fea_out_grad[:] = grad
        fea_model.backward([fea_out_grad])
        noise = fea_grad.asnumpy()

        for j in range(batch_size):
            if np.sum(np.abs(noise[j,:]))==0:
                noise[j,:] = 0
            else:
                noise[j,:] = (0.1 *np.linalg.norm(feature[j,:].flatten(),2) / np.linalg.norm(noise[j, :].flatten(), 2)) * noise[j, :]
        fea_arg_map["feature"][:] = feature + noise

        fea_model.forward(is_train=True)
        raw_output = fea_model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        grad = LogLossGrad(pred, label.asnumpy())
        fea_out_grad[:] = grad
        fea_model.backward([fea_out_grad])

        net_model.backward([fea_grad])
        for name in net_arg_names:
            if name != "data":
                if name.endswith("weight"):
                    SGD(net_arg_map[name], net_grad_map[name], lr)
                else:
                    SGD(net_arg_map[name], net_grad_map[name], lr, 0)
        for name in fea_arg_names:
            if name != "feature":
                if name.endswith("weight"):
                    SGD(fea_arg_map[name], fea_grad_map[name], lr)
                else:
                    SGD(fea_arg_map[name], fea_grad_map[name], lr, 0)
        nbatch += 1

    train_acc /= nbatch
    train_loss /= nbatch
    print("[%d] Train Accuracy: %.4f\t Train Loss: %.5f" % (i, train_acc, train_loss))


# In[ ]:
print('-' * 60)
print('Normal Validation: %.3f' % acc_normal(net_model, fea_model, val_iter, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('Fixed set perturbation: %.3f' % acc_normal(net_model, fea_model, perb_iter, net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('L0 perturbation: %.3f' % acc_perb_L0(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('L2 perturbation: %.3f' % acc_perb_L2(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
print('Alpha perturbation: %.3f' % acc_perb_alpha(net_model, fea_model, val_iter, 0.5,net_arg_map, net_grad_map, fea_arg_map, fea_grad_map))
"""
