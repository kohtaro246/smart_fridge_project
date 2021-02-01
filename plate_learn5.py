from PIL import Image
import numpy as np
import cv2
import glob
import pickle
import random
import matplotlib.pyplot as plt

random.seed(1)


def load_dataset():
    train_set = np.empty((1, 540//2, 960//2, 3))
    train_class = np.empty((1, 1), dtype='int64')
    test_set = np.empty((1, 540//2, 960//2, 3))
    test_class = np.empty((1, 1), dtype='int64')

    for i, data in enumerate(glob.glob("./data_minibatch/donburi/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        train_set = np.append(train_set, np.array([img]), axis=0)
        train_class = np.append(train_class, np.array([[0]]), axis=0)
        '''
        if i == 63:
            break
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/donburi_test/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        test_set = np.append(test_set, np.array([img]), axis=0)
        test_class = np.append(test_class, np.array([[0]]), axis=0)
        '''
        if i == 10:
            break
            print("here1")
        '''
    train_set = np.delete(train_set, 0, 0)
    test_set = np.delete(test_set, 0, 0)
    train_class = np.delete(train_class, 0, 0)
    test_class = np.delete(test_class, 0, 0)

    for i, data in enumerate(glob.glob("./data_minibatch/metal_cont/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        train_set = np.append(train_set, np.array([img]), axis=0)
        train_class = np.append(train_class, np.array([[1]]), axis=0)
        '''
        if i == 63:
            break
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/metal_cont_test/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        test_set = np.append(test_set, np.array([img]), axis=0)
        test_class = np.append(test_class, np.array([[1]]), axis=0)
        '''
        if i == 10:
            break
            print("here1")
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/smartsnap/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        train_set = np.append(train_set, np.array([img]), axis=0)
        train_class = np.append(train_class, np.array([[2]]), axis=0)
        '''
        if i == 63:
            break
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/smartsnap_test/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        test_set = np.append(test_set, np.array([img]), axis=0)
        test_class = np.append(test_class, np.array([[2]]), axis=0)
        '''
        if i == 10:
            break
            print("here1")
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/rice/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        train_set = np.append(train_set, np.array([img]), axis=0)
        train_class = np.append(train_class, np.array([[3]]), axis=0)
        '''
        if i == 63:
            break
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/rice_test/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        test_set = np.append(test_set, np.array([img]), axis=0)
        test_class = np.append(test_class, np.array([[3]]), axis=0)
        '''
        if i == 10:
            break
            print("here1")
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/soup_s/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        train_set = np.append(train_set, np.array([img]), axis=0)
        train_class = np.append(train_class, np.array([[4]]), axis=0)
        '''
        if i == 63:
            break
        '''

    for i, data in enumerate(glob.glob("./data_minibatch/soup_s_test/*")):
        img = cv2.imread(data)
        img = cv2.resize(img, (960//2, 540//2))
        test_set = np.append(test_set, np.array([img]), axis=0)
        test_class = np.append(test_class, np.array([[4]]), axis=0)
        '''
        if i == 10:
            break
            print("here1")
        '''

    return train_set, test_set, np.transpose(train_class), np.transpose(test_class)


train_set, test_set, train_class, test_class = load_dataset()
print("data loaded")
# print(train_set.shape)
print(train_class.shape)
# print(test_set.shape)
# print(test_class.dtype)

import cupy as np
# add here "import cupy as np" to use gpu
train_set = np.asarray(train_set)
test_set = np.asarray(test_set)
train_class = np.asarray(train_class)
test_class = np.asarray(test_class)

# flatten and normalize
train_set = train_set.reshape(train_set.shape[0], -1).T / 255.0
test_set = test_set.reshape(test_set.shape[0], -1).T / 255.0
# print(train_set.shape)

# convert to onehot
train_class = np.eye(5)[train_class.reshape(-1)].T
test_class = np.eye(5)[test_class.reshape(-1)].T
# print(test_class.shape)

# create 4 minibatches


def create_random_minibatches(batch_num, train_set, train_class, seed):
    assert train_set.shape[1] % batch_num == 0
    train_sets = []
    train_classes = []
    batch_train_set_len = train_set.shape[1] // batch_num

    perm = np.random.permutation(train_set.shape[1]).tolist()
    train_set_shuffled = train_set[:, perm]
    train_class_shuffled = train_class[:, perm]

    for i in range(0, batch_num):
        train_sets.append(
            train_set_shuffled[:, (i*batch_train_set_len):((i+1)*batch_train_set_len)])
        train_classes.append(
            train_class_shuffled[:, (i*batch_train_set_len):((i+1)*batch_train_set_len)])

    return train_sets, train_classes


def init_params(dims):

    params = {}

    for i in range(1, len(dims)):
        params['w'+str(i)] = np.random.randn(dims[i], dims[i-1]) * 0.01
        params['b'+str(i)] = np.zeros((dims[i], 1))

    return params


#print(init_params([train_set.shape[0], 25, 12, 6])["b2"].shape)
# print((np.dot(init_params([train_set.shape[0], 25, 12, 6])[
#      "w1"], test_set)+init_params([train_set.shape[0], 25, 12, 6])[
#    "b1"]).shape)
# define activation functions


def relu(z):
    a = np.maximum(0, z)
    return a


def softmax(z):
    a = np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)
    return a


def for_prop_step(x, w, b, activation_type):
    z = np.dot(w, x)+b

    if activation_type == "relu":
        a = relu(z)
        # print(a.shape)
    elif activation_type == "softmax":
        a = softmax(z)

    return z, a


def for_prop(inp, params):
    layer_num = len(params)//2
    tmps = []
    x = inp
    # print(x.shape)
    for i in range(1, layer_num):
        z, a = for_prop_step(x, params['w'+str(i)], params['b'+str(i)], "relu")
        tmps.append([z, a, params['w'+str(i)], params['b'+str(i)], x])
        x = a
        # print(z.shape)

    z, a = for_prop_step(
        x, params['w'+str(layer_num)], params['b'+str(layer_num)], "softmax")
    tmps.append([z, a, params['w'+str(layer_num)],
                 params['b'+str(layer_num)], x])

    return a, tmps


# a, tmps = for_prop(train_set, init_params([train_set.shape[0], 25, 12, 5]))
# print(a)
# tmps return the following [z,a,wi,bi,xi]


def ce_loss_l2(a, label, params, lamb):
    layer_num = len(params) // 2
    cost_tmp = -np.sum(label*np.log(a), axis=0, keepdims=True)
    # print(cost_tmp.shape[1])
    cost = np.sum(cost_tmp)/cost_tmp.shape[1]

    l2_reg_cost = 0.0
    for i in range(0, layer_num):
        l2_norm = np.linalg.norm(params["w" + str(i+1)], "fro")
        l2_reg_cost += l2_norm ** 2

    l2_reg_cost = l2_reg_cost * lamb / 2 / cost_tmp.shape[1]
    cost = cost + l2_reg_cost
    return cost


#print(ce_loss_l2(a, train_class))


def back_prop_step(da, tmp, activation_type, lamb, minibatch_class):
    train_class = minibatch_class
    z = np.array(tmp[0], copy=True)
    a = np.array(tmp[1], copy=True)
    w = np.array(tmp[2], copy=True)
    b = np.array(tmp[3], copy=True)

    x = np.array(tmp[4], copy=True)

    if activation_type == "relu":
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
    elif activation_type == "softmax":
        dz = a-train_class

    dw = np.dot(dz, np.transpose(x)) / \
        train_class.shape[0] + w * lamb / train_class.shape[0]
    db = np.sum(dz, axis=1, keepdims=True) / train_class.shape[0]
    dx = np.dot(np.transpose(w), dz)
    # print("w")
    # print(w.shape)
    # print("dw")
    # print(dw.shape)
    # print(x.shape)
    return dx, dw, db


# print(tmps[-1][1])


def back_prop(tmps, minibatch_class):
    derivs = {}
    layer_num = len(tmps)

    derivs["da" + str(layer_num-1)], derivs["dw" + str(layer_num)], derivs["db" +
                                                                           str(layer_num)] = back_prop_step(None, tmps[-1], "softmax", 1.0, minibatch_class)
    for i in reversed(range(layer_num-1)):
        da_prev, dw, db = back_prop_step(
            derivs["da" + str(i+1)], tmps[i], "relu", 1.0, minibatch_class)
        derivs["da" + str(i)] = da_prev
        derivs["dw" + str(i+1)] = dw
        derivs["db" + str(i+1)] = db

    return derivs


# print(back_prop(tmps))

def update(params, derivs, learning_rate):
    layer_num = len(params) // 2

    for i in range(layer_num):
        params["w" + str(i+1)] = params["w" + str(i+1)] - \
            learning_rate * derivs["dw" + str(i+1)]
        params["b" + str(i+1)] = params["b" + str(i+1)] - \
            learning_rate * derivs["db" + str(i+1)]

    return params


def learn(train_set, train_class, learning_rate, batch_num, epochs):
    cost = 0
    cost_prev = 0
    cost_prev_prev = 0
    train_accu = 0.0
    test_accu = 0.0
    global test_set
    global test_class
    costs = []
    train_accuracies = []
    test_accuracies = []

    # initialize params
    params = init_params([train_set.shape[0], 25, 12, 5])

    for i in range(0, epochs):
        seed = 40
        train_sets, train_classes = create_random_minibatches(
            batch_num, train_set, train_class, seed)
        j = 0
        for minibatch in train_sets:
            a, tmps = for_prop(minibatch, params)
            cost_prev_prev = cost_prev
            cost_prev = cost
            cost = ce_loss_l2(a, train_classes[j], params, 1.0)
            
            if cost < 0.2 or test_accu > 0.8:
                print("Cost after epoch {} : {}" .format(i, np.squeeze(cost)))
                print("learning rate: {}" .format(learning_rate))
                break
            if cost_prev_prev < cost_prev and cost_prev < cost:
                learning_rate = learning_rate * 1
            derivs = back_prop(tmps, train_classes[j])
            params = update(params, derivs, learning_rate)
            j += 1
        costs.append(cost)
            
        print("Cost after epoch {} : {}" .format(i, np.squeeze(cost)))
        print("learning rate: {}" .format(learning_rate))
        
        if i % 1000 == 0:
            prob1, train_accu = accuracy(train_set, train_class, params)
            prob2, test_accu = accuracy(test_set, test_class, params)
            train_accuracies.append(train_accu)
            test_accuracies.append(test_accu)
            print("Train accuracy: {}" .format(train_accu))
            print("Test accuracy: {}" .format(test_accu))
            
        seed += 1
        # print(i)

    return params, costs, train_accuracies, test_accuracies


def accuracy(data, label, params):
    # print(data.shape[1])
    m = data.shape[1]
    n = len(params) // 2
    prob, tmps = for_prop(data, params)
    #print(prob[:, 0])
    types = prob[:, 0].shape[0]
    # print(types)
    for i in range(0, prob.shape[1]):
        m_index = np.argmax(prob[:, i])
        # print(m_index)
        for j in range(0, types):
            if j == m_index:
                prob[j][i] = 1
                # print("prob")
                # print(prob[j][i])
            else:
                prob[j][i] = 0

    sum = 0.0
    for i in range(0, m):

        #print("prob: {}".format(prob[:, i]))
        #print("label: {}".format(label[:, i]))
        if (prob[:, i] == label[:, i]).all():
            sum += 1.0

    #print("Accuracy:" + str(sum/float(m)))
    return prob, sum/float(m)


#print("learning rate = 0.0005")
#params = learn(train_set, train_class, 0.0005, 3500)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.0004")
#learn(train_set, train_class, 0.0004, 3500)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.0003")
#learn(train_set, train_class, 0.0003, 4000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.0002")
#learn(train_set, train_class, 0.0002, 4000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
print("learning rate = 0.0002")
params, costs, train_accuracies, test_accuracies = learn(
    train_set, train_class, 0.0002, 4, 1500)
prob1, train_accu = accuracy(train_set, train_class, params)
prob2, test_accu = accuracy(test_set, test_class, params)
print("Train accuracy: {}" .format(train_accu))
print("Test accuracy: {}" .format(test_accu))
with open('./params.pickle', mode='wb') as f:
    pickle.dump(params, f)
fig = plt.figure()
plt.plot(costs, label="cost")
plt.title('cost')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(loc="lower right")
fig.savefig("./result/cost.png")
# plt.show()

fig = plt.figure()
plt.plot(train_accuracies, label="train_accuracy")
plt.plot(test_accuracies, label="test_accuracy")
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc="lower right")
fig.savefig("./result/accuracy.png")
# plt.show()
#print("learning rate = 0.00005")
#learn(train_set, train_class, 0.00005, 6000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.00001")
#learn(train_set, train_class, 0.00001, 10000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
