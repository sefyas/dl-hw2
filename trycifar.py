from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
            #make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, batchnorm=1), # YSS
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
            #make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, batchnorm=1), # YSS
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
            #make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, batchnorm=1), # YSS
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
            #make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, batchnorm=1), # YSS
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = 0.1 # rate = .01 # YSS
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

# 8*27*1024 + 16*72*256 + 32*144*64 + 64*288*16 + 256*10
# 1108480
# 221696
# 3072 input
# 72 out

# YSS 
# --- Comparison of performance with/out batchnorm ---
# Without batchnorm (learning rate = 0.01)
#    - training accuracy: 71%
#    - test accuracy:     65%
# With batchnorm (learning rate = 0.01)
#    - training accuracy: 77%
#    - test accuracy:     70%
# batch normalization has clearly helped the network learn and generalize better
# 
# --- Training with large bur increasinly smaller learning rates ---
# starting with rate = 0.1 and scaling it by half after each 1000 iterations
# With batchnorm
#    - training accuracy: 68%
#    - test accuracy:     66%
# it is interesting that 