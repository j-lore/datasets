import ktc
import os
import onnx
from PIL import Image
import numpy as np

###  post process function  ###
import tensorflow as tf
import pathlib
import sys
sys.path.append(str(pathlib.Path("keras_yolo3").resolve()))
from yolo3.model import yolo_eval

def postprocess(inf_results, ori_image_shape):
    tensor_data = [tf.convert_to_tensor(data, dtype=tf.float32) for data in inf_results]

    # get anchor info
    anchors_path = "/data1/keras_yolo3/model_data/tiny_yolo_anchors.txt"
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

    # post process
    num_classes = 80
    boxes, scores, classes = yolo_eval(tensor_data, anchors, num_classes, ori_image_shape)
    with tf.Session() as sess:
        boxes = boxes.eval()
        scores = scores.eval()
        classes = classes.eval()

    return boxes, scores, classes

###  pre process function  ###
from yolo3.utils import letterbox_image

def preprocess(pil_img):
    model_input_size = (416, 416)  # to match our model input size when converting
    boxed_image = letterbox_image(pil_img, model_input_size)
    np_data = np.array(boxed_image, dtype='float32')
    # Reshape the input data to match the model's expected shape
    np_data = np_data.transpose((2, 0, 1))  # Change channel order to (3, 416, 416)
    np_data = np.reshape(np_data, (1, 3, 416, 416))

    # change normalization method due to we add "pixel_modify" BN node at model's front
    np_data /= 255.
    np_data -= 128
    
    return np_data
  
# convert h5 model to onnx
m = ktc.onnx_optimizer.keras2onnx_flow("/data1/yolo.h5", input_shape = [1,416,416,3])
m = ktc.onnx_optimizer.onnx2onnx_flow(m)

# add pixel modify node:
#   1. scaling 1/255 for every channel due to original normalize method, 
#   2. shift 0.5 to change input range from 0~255 to -128 to 127
ktc.onnx_optimizer.pixel_modify(m,[1/255,1/255,1/255],[0.5,0.5,0.5])

# do onnx2onnx again to calculate "pixel_modify" BN node's output shape
m = ktc.onnx_optimizer.onnx2onnx_flow(m)

onnx.save(m,'yolo.opt.onnx')

# onnx_file = "/data1/yolo.opt.onnx"
# m = onnx_file
# setup ktc config
km = ktc.ModelConfig(85, "0001", "720", onnx_model = m)

# npu(only) performance simulation
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + str(eval_result))


## onnx model check
input_image = Image.open('/data1/000000350003.jpg')
in_data = preprocess(input_image)
out_data = ktc.kneron_inference([in_data], onnx_file="/data1/yolo.opt.onnx", input_names=["input_1_o0"])
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)

# load and normalize all image data from folder
img_list = []
for (dir_path, _, file_names) in os.walk("/data1/test_image10"):
    for f_n in file_names:
        fullpath = os.path.join(dir_path, f_n)
        print("processing image: " + fullpath)

        image = Image.open(fullpath)
        img_data = preprocess(image)
        img_list.append(img_data)


# fix point analysis
bie_model_path = km.analysis({"input_1_o0": img_list})
print("\nFix point analysis done. Save bie model to '" + str(bie_model_path) + "'")


# bie model check
input_image = Image.open('/data1/000000350003.jpg')
in_data = preprocess(input_image)
radix = ktc.get_radix(img_list)
# out_data = ktc.kneron_inference([in_data], bie_file=bie_model_path, input_names=["input_1_o0"], radix=radix, platform=530)
# det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
# print(det_res)


# compile
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")

# nef model check
input_image = Image.open('/data1/000000350003.jpg')
in_data = preprocess(input_image)
radix = ktc.get_radix(img_list)
# out_data = ktc.kneron_inference([in_data], nef_file=nef_model_path, radix=radix, platform=530)
# det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
# print(det_res)