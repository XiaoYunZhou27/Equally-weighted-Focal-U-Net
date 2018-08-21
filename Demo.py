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

from __future__ import division, print_function
from tf_unet import unet, image_util, util
import numpy as np
import os
import h5py
import scipy.io as sio

# Train
os.system('rm -rf /data/XIAOYUN_ZHOU/Marker_Seg/Trained_1/prediction/*') # remove the saved predictions in your last training
os.system('rm -rf /data/XIAOYUN_ZHOU/Marker_Seg/Trained_1/parameter/*') # remove the saved models in your last training
Unet_path = "/data/XIAOYUN_ZHOU/Marker_Seg/Trained_1/parameter/" # specify the saving path for trained models
Data_path = "/data/XIAOYUN_ZHOU/CodeRelease/IROS2018/Data/Train/" # specify the path for training images
Train_num = 80*72
Veri_num = 7*72

net = unet.Unet(channels=1, n_class=6, layers=3, features_root=64,
                cost_kwargs=dict(fore_weights=1.0, back_weights=1.0))

trainer = unet.Trainer(net, optimizer="momentum",
                       opt_kwargs=dict(momentum=0.9,
                                       learning_rate_step=[10000000],
                                       learning_rate_value=[0.01, 0.1]))

path = trainer.train(Unet_path, Data_path, Train_num, Veri_num,
                     training_iters=200, epochs=50000, restore=False)

# # Test
# Save_path = '/data/XIAOYUN_ZHOU/Marker_Seg/Test/result/' # please specify this file for saveing results
# Data_path = "/data/XIAOYUN_ZHOU/CodeRelease/IROS2018/Data/" # please specify this file to your test data file
# # Load testing data
# net = unet.Unet(channels=1, n_class=6, layers=3, features_root=64,
#                 cost_kwargs=dict(fore_weights=1.0, back_weights=1.0))
# Image_size = 512
# N_class = 6
# Image_t = sio.loadmat(Data_path + "Marker_image_test_normalized.mat")
# Image_test = Image_t['Marker_image_test_normalized']
# Label_t = sio.loadmat(Data_path + "Marker_label_test_multipleclass.mat")
# Label_test = Label_t['Marker_label_test_multipleclass']
#
# prediction = np.zeros((Image_test.shape[0], Image_size, Image_size, N_class))
# img = np.zeros((Image_size, Image_size*3,3,N_class))
#
# for i in range(0, np.shape(Image_test)[0]):
#     x_test = np.reshape(Image_test[i, ...], (1, Image_size, Image_size, 1))
#     pred = net.predict(x_test)
#     y_test = np.reshape(Label_test[i, ...], (1, Image_size, Image_size, N_class))
#     prediction[i, ...] = pred
#     img[...,0],img[...,1],img[...,2],img[...,3],img[...,4],img[...,5]=util.combine_img_prediction(x_test,y_test,pred)
#     for j in range(0,6):
#         util.save_image(img[...,j], (Save_path+"%s_%s.jpg"%(i+1,j)))
# Marker_Seg = {}
# Marker_Seg['Marker_Seg'] = prediction
# sio.savemat(Save_path+'Marker_Seg.mat', Marker_Seg)