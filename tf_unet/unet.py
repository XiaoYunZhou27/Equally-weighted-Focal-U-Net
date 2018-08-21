# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# this code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import scipy.io as sio
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from tf_unet import util
from tf_unet.layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
                            cross_entropy)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        
        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size -= 4
        if layer < layers-1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2
        
    in_node = dw_h_convs[layers-1]
        
    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size *= 2
        size -= 4

    # Output Map
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map
    
    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1))
            tf.summary.image('summary_conv_%02d_02'%i, get_image_summary(c2))
            
        for k in pools.keys():
            tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))
        
        for k in deconv.keys():
            tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k]))
            
        for k in dw_h_convs.keys():
            tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

        for k in up_h_convs.keys():
            tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])
            
    variables = []
    for w1,w2 in weights:
        variables.append(w1)
        variables.append(w2)
        
    for b1,b2 in biases:
        variables.append(b1)
        variables.append(b2)

    
    return output_map, variables, int(in_size - size)


class Unet(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """
    
    def __init__(self, channels=3, n_class=2, cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()
        
        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        
        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        logits, self.variables, self.offset = create_conv_net(self.x, self.keep_prob, channels, n_class, **kwargs)
        
        self.cost = self._get_cost(logits, cost, cost_kwargs)
        
        self.gradients_node = tf.gradients(self.cost, self.variables)
         
        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))
        
        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
        
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            fore_weights = cost_kwargs.pop("fore_weights", None)
            back_weights = cost_kwargs.pop("back_weights", None)
            
            if fore_weights is not None:

                # By XY

                logits_softmax = tf.nn.softmax(flat_logits)
                weight_fore = tf.constant(np.array(fore_weights, dtype=np.float32))
                weight_back = tf.constant(np.array(back_weights, dtype=np.float32))
                weight_map_fore = tf.multiply(flat_labels, weight_fore)
                weight_map_back = tf.multiply(flat_labels, weight_back)

                # Weighted loss - use this loss at the first-step training
                weight_loss = -weight_map_fore[..., 0] * tf.log(logits_softmax[..., 0])
                for i_map in range(1, self.n_class):
                    weight_loss = weight_loss-weight_map_back[..., i_map]*tf.log(logits_softmax[..., i_map])
                loss = tf.reduce_mean(weight_loss)

                # # Focal loss - use this loss at the second-step training
                # focal_map = tf.ones(tf.shape(logits_softmax), tf.float32) - logits_softmax
                # focal_map_2 = tf.multiply(focal_map, focal_map)
                # focal_loss = -weight_map_fore[..., 0]*focal_map_2[..., 0]*tf.log(logits_softmax[..., 0])# weighted background
                # for i_map in range(1, self.n_class):
                #     focal_loss = focal_loss-weight_map_back[..., i_map]*focal_map_2[..., i_map]*tf.log(logits_softmax[..., i_map])
                # loss = tf.reduce_mean(focal_loss)

                # By XY
                
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection/ (union))
            
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)
            
        return loss

    # def predict(self, model_path, x_test):
    # By XY
    def predict(self, x_test):
    # By XY
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            # self.restore(sess, model_path)
            # By XY
            Restore_path = "/data/XIAOYUN_ZHOU/CodeRelease/IROS2018/TrainedModels" # please provede this path for checkpoint restore
            ckpt = tf.train.get_checkpoint_state(Restore_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.restore(sess, ckpt.model_checkpoint_path)
            # By XY
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
            
        return prediction
    
    # def save(self, sess, model_path):
    # By XY
    def save(self, sess, model_path, save_step):
    # By XY
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        # save_path = saver.save(sess, model_path)
        # By XY
        save_path = saver.save(sess, model_path, global_step=save_step)
        # By XY
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    """
    
    # prediction_path = "prediction"
    # By XY
    prediction_path = "/data/XIAOYUN_ZHOU/Marker_Seg/Trained_1/prediction/"
    # By XY
    verification_batch_size = 4
    
    def __init__(self, net, batch_size=1, norm_grads=False, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            # By XY
            learning_rate_step = self.opt_kwargs.pop("learning_rate_step", [100000])
            learning_rate_value = self.opt_kwargs.pop("learning_rate_value", [0.01, 0.001])
            # By XY
            
            # self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
            #                                             global_step=global_step,
            #                                             decay_steps=training_iters,
            #                                             decay_rate=decay_rate,
            #                                             staircase=True)
            # By XY
            self.learning_rate_node = tf.train.piecewise_constant(x=global_step,
                                                                  boundaries=learning_rate_step,
                                                                  values=learning_rate_value)
            # By XY
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost, 
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore):
        global_step = tf.Variable(0)
        
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))
        
        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        
        prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(prediction_path))
            shutil.rmtree(prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(prediction_path):
            logging.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init

    # def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False, write_graph=False):
    # By XY
    def train(self, Unet_path, Data_path, Train_num, Veri_num,
              training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False, write_graph=False):
    # By XY
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        # save_path = os.path.join(output_path, "model.cpkt")
        # if epochs == 0:
        #     return save_path
        # By XY
        output_path = Unet_path;
        Initial_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return Initial_path
        # By XY
        
        init = self._initialize(training_iters, output_path, restore)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            
            if restore:
                # ckpt = tf.train.get_checkpoint_state(output_path)
                # By XY
                Restore_path = "/data/XIAOYUN_ZHOU/Marker_Seg/Trained_1/restore" # please provide this path for checkpoint restore
                ckpt = tf.train.get_checkpoint_state(Restore_path)
                # By XY
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            # test_x, test_y = data_provider(self.verification_batch_size)
            # pred_shape = self.store_prediction(sess, test_x, test_y, "_init")
            # By XY
            # test_x, test_y = Veri_data(self.batch_size)
            idx = np.random.choice(Veri_num)+1
            Image_t = sio.loadmat(Data_path + "Marker_image_verification_augment_%s.mat"%(idx))
            test_x = Image_t['Marker_image_verification_augment']
            test_x = np.reshape(test_x, (1, test_x.shape[0], test_x.shape[1], 1))
            Label_t = sio.loadmat(Data_path + "Marker_label_verification_multipleclass_augment_%s.mat"%(idx))
            test_y = Label_t['Marker_label_verification_multipleclass_augment']
            test_y = np.reshape(test_y, (1, test_y.shape[0], test_y.shape[1], test_y.shape[2]))
            pred_shape, _ = self.store_prediction(sess, test_x, test_y, "_init")
            # By XY
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")
            
            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    # batch_x, batch_y = data_provider(self.batch_size)
                    idx = np.random.choice(Train_num)+1
                    Image_t = sio.loadmat(Data_path + "Marker_image_train_augment_%s.mat" % (idx))
                    batch_x = Image_t['Marker_image_train_augment']
                    batch_x = np.reshape(batch_x, (1, batch_x.shape[0], batch_x.shape[1], 1))
                    Label_t = sio.loadmat(Data_path + "Marker_label_train_multipleclass_augment_%s.mat" % (idx))
                    batch_y = Label_t['Marker_label_train_multipleclass_augment']
                    batch_y = np.reshape(batch_y, (1, batch_y.shape[0], batch_y.shape[1], batch_y.shape[2]))
                     
                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node), 
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                                 self.net.keep_prob: dropout})

                    if self.net.summaries and self.norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()
                    
                    # if step % display_step == 0:
                    #     self.output_minibatch_stats(sess, summary_writer, step, batch_x, util.crop_to_shape(batch_y, pred_shape))
                        
                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                # self.store_prediction(sess, test_x, test_y, "epoch_%s" % epoch)
                # By XY
                # test_x_tmp, test_y_tmp = Veri_data(self.batch_size)
                idx = np.random.choice(Veri_num)+1
                Image_t = sio.loadmat(Data_path + "Marker_image_verification_augment_%s.mat" % (idx))
                test_x_tmp = Image_t['Marker_image_verification_augment']
                test_x_tmp = np.reshape(test_x_tmp, (1, test_x_tmp.shape[0], test_x_tmp.shape[1], 1))
                Label_t = sio.loadmat(Data_path + "Marker_label_verification_multipleclass_augment_%s.mat" % (idx))
                test_y_tmp = Label_t['Marker_label_verification_multipleclass_augment']
                test_y_tmp = np.reshape(test_y_tmp, (1, test_y_tmp.shape[0], test_y_tmp.shape[1], test_y_tmp.shape[2]))
                _, prediction_tmp = self.store_prediction(sess, test_x_tmp, test_y_tmp, "epoch_%s"%epoch)
                # for i_tmp in range(0, 6):
                #     print(np.amin(prediction_tmp[..., i_tmp]))
                # By XY
                    
                # save_path = self.net.save(sess, save_path)
                # By XY
                save_path = self.net.save(sess, Initial_path, epoch)
                # By XY
            logging.info("Optimization Finished!")
            
            return save_path
        
    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, 
                                                             self.net.y: batch_y, 
                                                             self.net.keep_prob: 1.})
        pred_shape = prediction.shape
        
        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x, 
                                                       self.net.y: util.crop_to_shape(batch_y, pred_shape), 
                                                       self.net.keep_prob: 1.})
        
        logging.info("Verification loss= {:.4f}".format(loss))
              
        # img = util.combine_img_prediction(batch_x, batch_y, prediction)
        # util.save_image(img, "%s/%s.jpg"%(self.prediction_path, name))
        img_0, img_1, img_2, img_3, img_4, img_5 = util.combine_img_prediction(batch_x, batch_y, prediction)
        util.save_image(img_0, "%s/%s_1.jpg" % (self.prediction_path, name))
        util.save_image(img_1, "%s/%s_2.jpg" % (self.prediction_path, name))
        util.save_image(img_2, "%s/%s_3.jpg" % (self.prediction_path, name))
        util.save_image(img_3, "%s/%s_4.jpg" % (self.prediction_path, name))
        util.save_image(img_4, "%s/%s_5.jpg" % (self.prediction_path, name))
        util.save_image(img_5, "%s/%s_6.jpg" % (self.prediction_path, name))
        
        # return pred_shape
        # By XY
        return pred_shape, prediction
        # By XY
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        # logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
        # By XY
        logging.info("Epoch {:}, learning rate: {:.8f}, Average loss: {:.12f},".format(epoch, lr, (total_loss / training_iters)))
        # By XY
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op, 
                                                            self.net.cost, 
                                                            self.net.accuracy, 
                                                            self.net.predicter], 
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                            loss,
                                                                                                            acc,
                                                                                                            error_rate(predictions, batch_y)))

def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
        
    return avg_gradients

def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V