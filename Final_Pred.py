'''
Title           :Predictions_Final.py
Description     :Makes prediction for DeepLearning_crop_classification.
Author          :Nitin Shukla
Date Created    :20170707
'''

import os
import glob
import sys
import requests
import json
sys.path.append('/home/ubuntu/caffe/python')

import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

from flask import Flask, render_template, request, redirect

app = Flask(__name__)

caffe.set_mode_cpu()

#Spatial dimension
IMG_WIDTH = 224
IMG_HEIGHT = 224

#CAFFE NET DEFINITON
#Read model architecture and trained model's weights
net_crop = caffe.Net('/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/caffe_models/Resnet_32/deploy_32.prototxt',
                '/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/caffe_models/Resnet_32_iter_10000.caffemodel',
                caffe.TEST)
net_leaves = caffe.Net('/home/ubuntu/DeepLearning_crop_classification/Caffe_Model_Leaves/Resnet_32/deploy_32.prototxt',
                '/home/ubuntu/DeepLearning_crop_classification/Caffe_Model_Leaves/Resnet_32_iter_5000.caffemodel',
                caffe.TEST)
net_plant = caffe.Net('/home/ubuntu/DeepLearning_crop_classification/Caffe_Model_Plant/Resnet_32/deploy.prototxt',
                '/home/ubuntu/DeepLearning_crop_classification/Caffe_Model_Plant/Resnet_32_iter_7000.caffemodel',
                caffe.TEST)
net_plot = caffe.Net('/home/ubuntu/DeepLearning_crop_classification/Caffe_Model_Plot/Resnet_32/deploy_32.prototxt',
                '/home/ubuntu/DeepLearning_crop_classification/Caffe_Model_Plot/Resnet_32_iter_4000.caffemodel',
                caffe.TEST)

'''
Reading mean image, caffe model and its weights
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()

# Crop_Classification
with open('/home/ubuntu/DeepLearning_crop_classification/input_2/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array_crop = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

# Leaves
with open('/home/ubuntu/DeepLearning_crop_classification/input/LEAVES/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array_leaves = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

# Plant
with open('/home/ubuntu/DeepLearning_crop_classification/input/PLANT/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array_plant = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

# Plot
with open('/home/ubuntu/DeepLearning_crop_classification/input/PLOT/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array_plot = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))
'''
Processing
'''

@app.route("/", methods=['GET'])
def main():
    #Define image transformers
    print "Logged here"
    # Crop_Classification
    transformer_crop = caffe.io.Transformer({'data': net_crop.blobs['data'].data.shape})
    transformer_crop.set_mean('data', mean_array_crop)
    transformer_crop.set_transpose('data', (2,0,1))

    # Leaves
    transformer_leaves = caffe.io.Transformer({'data': net_leaves.blobs['data'].data.shape})
    transformer_leaves.set_mean('data', mean_array_leaves)
    transformer_leaves.set_transpose('data', (2,0,1))

    # Plant
    transformer_plant = caffe.io.Transformer({'data': net_plant.blobs['data'].data.shape})
    transformer_plant.set_mean('data', mean_array_plant)
    transformer_plant.set_transpose('data', (2,0,1))

    # Plot
    transformer_plot = caffe.io.Transformer({'data': net_plot.blobs['data'].data.shape})
    transformer_plot.set_mean('data', mean_array_plot)
    transformer_plot.set_transpose('data', (2,0,1))

    '''
    Making predicitions
    '''
    #Reading image paths
    image_name = request.args.get('image_name')
    url = request.args.get('url')
    #url = 'http://thorium.prakshep.com/img/prakshep_admin.jpg'
    #image_name = 'prakshep_admin.jpg'
    i = 0
    while download_image(url,image_name) == 'false' :
        i = i +1 ;
        if(i == 5) :
            print "Failed after 5 tries"
            data = {"status":"500","message":"Image Download failed"}
            return json.dumps(data)

    test_img_paths = [os.path.join("images/", image_name)]
    print test_img_paths, "print test img paths"
   # Making predictions

   f_pred = []   # final_prediction
   for img_path in test_img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMG_WIDTH, img_height = IMG_HEIGHT)

        net_crop.blobs['data'].data[...] = transformer_crop.preprocess('data', img)
        out_crop = net_crop.forward()
        pred_probas_crop = out_crop['prob']

        # test_ids = img_path.split('/')[-1][:-4]
        preds_crop = pred_probas_crop.argmax()
        f_pred.append(preds_crop)

        # Leaves
        if preds_crop ==0:
            net_leaves.blobs['data'].data[...] = transformer_leaves.preprocess('data', img)
            out_leaves = net_leaves.forward()
            pred_probas_leaves = out_leaves['prob']

            # test_ids = img_path.split('/')[-1][:-4]
            preds = pred_probas_leaves.argmax()

        # Plant
        elif pred_crop ==1:
            net_plant.blobs['data'].data[...] = transformer_plant.preprocess('data', img)
            out_plant = net_plant.forward()
            pred_probas_plant = out_plant['prob']

            # test_ids = img_path.split('/')[-1][:-4]
            preds = pred_probas_plant.argmax()

        # Plot
        else :
            net_plot.blobs['data'].data[...] = transformer_plot.preprocess('data', img)
            out_plot = net_plot.forward()
            pred_probas_plot = out_plot['prob']

            # test_ids = img_path.split('/')[-1][:-4]
            preds = pred_probas_plot.argmax()

        f_pred.append(preds)


    print preds_crop, preds
	print img_path,"img_path"
       # print preds
        print '-------'
        return json.dumps({"type":f_pred})

def transform_img(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    print "transform_img function called"
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def download_image(url,image_name):
    print(url)
    try:
        image_path = os.path.join("images/", image_name)
        r = requests.get(url)
        with open(image_path, "wb") as code:
            code.write(r.content)
    except ValueError :
        print("Invalid URL !")
        return "false"
    except :
        print(sys.exc_info())
        print("Unknown Exception" + str(sys.exc_info()[0]))
        return "false"
    return "true"

if __name__ == "__main__" :
    app.run(host="0.0.0.0",port=5000)
