import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.summary import summary
import os
#os.system('taskset -p 0xffffffff %d' % os.getpid())
# For MobileNet Quant
def dic_gen(name): #For loading from pb file
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    dic = {}
    if name != 'FC':
        dic['array'] = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/MobilenetV1/' + name + '/weights_quant/FakeQuantWithMinMaxVars:0').eval()
        a_min = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/MobilenetV1/' + name + '/weights_quant/min/read:0').eval()
        a_max = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/MobilenetV1/' + name + '/weights_quant/max/read:0').eval()
        scale = (a_max - a_min)/255
    elif name == 'FC':
        dic['array'] = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/Logits/Conv2d_1c_1x1/weights_quant/FakeQuantWithMinMaxVars:0').eval()
        a_min = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/Logits/Conv2d_1c_1x1/weights_quant/min/read:0').eval()
        a_max = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/Logits/Conv2d_1c_1x1/weights_quant/max/read:0').eval()
        scale = (a_max - a_min)/255
    else:
        print('error')

    dic['array'] = np.floor(dic['array']/scale).astype('float')
    offset = 0
    dic['quantization'] = [scale, offset]
    return dic, scale
    
def dic_gen_bias(name, weight_scale):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    Relu6_scale = 5.999761581420898/255
    scale = weight_scale*Relu6_scale
    dic = {}
    if name =='FC':
        dic['array'] = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/Logits/Conv2d_1c_1x1/biases/read:0').eval()
    else:
        dic['array'] = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/MobilenetV1/' + name + '/BatchNorm_Fold/bias:0').eval()
    dic['array'] = dic['array']/scale
    offset = 0
    dic['quantization'] = [scale, offset]
    return dic
    
def load_from_pb():    #symmatric
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    model_dir = '../mobilenet_v1_1.0_224_quant/mobilenet_v1_1.0_224_quant_frozen.pb'

    parameters = {}
    tensor_dic = {}
    dic = {}
    with tf.Session() as sess:
        with gfile.GFile(model_dir, "rb") as f:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(f.read())
            importer.import_graph_def(graph_def)

        ops = tf.get_default_graph().get_operations()

        # a = tf.get_default_graph().get_tensor_by_name('import/MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars:0').eval()
        # print(a)

        #Input quantization
        dic['quantization'] = [1/255, 0]
        tensor_dic['input'] = dic

        #Conv2D
        name = 'Conv2d_0'
        tensor_dic[name + '/weights'], scale = dic_gen(name)
        tensor_dic[name + '/bias'] = dic_gen_bias(name, scale)

        #dw+pw
        for i in range(1,14):
            #dw
            name = 'Conv2d_' + str(i) + '_depthwise'
            tensor_dic[name + '/weights'], scale = dic_gen(name)
            tensor_dic[name + '/bias'] = dic_gen_bias(name, scale)
            #pw
            name = 'Conv2d_' + str(i) + '_pointwise'
            tensor_dic[name + '/weights'], scale = dic_gen(name)
            tensor_dic[name + '/bias'] = dic_gen_bias(name, scale)
        #FC
        name = 'FC'
        tensor_dic[name + '/weights'], scale = dic_gen(name)
        tensor_dic[name + '/bias'] = dic_gen_bias(name, scale)
    return tensor_dic

def load_from_tflite(quant_type):
    #os.system('taskset -p 0xffffffff %d' % os.getpid())
    if quant_type != 'uint' and quant_type != 'int':
        print('Quant type Error')

    # Load TFLite model and allocate tensors.
    interpreter = tf.contrib.lite.Interpreter(model_path='../mobilenet_v1_1.0_224_quant/mobilenet_v1_1.0_224_quant.tflite')
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()

    tensor_dic = {}
    if_array = 1
    if_valid = 1
    for dic in tensor_details:
        if_array = 1
        if_valid = 1
        tensor_name = dic['name']
        if tensor_name.endswith('input'):
            tensor_name = 'input'
            if_array = 0
        elif tensor_name.endswith('Conv2d_1c_1x1/weights_quant/FakeQuantWithMinMaxVars'):
            tensor_name = 'FC/weights'
            dic['array'] = interpreter.get_tensor(dic['index'])
            dic['array'] = np.moveaxis(dic['array'], 0, -1)
        elif tensor_name.endswith('Conv2d_1c_1x1/Conv2D_bias'):
            tensor_name = 'FC/bias'
            dic['array'] = interpreter.get_tensor(dic['index'])
        elif tensor_name.endswith('Conv2d_0/weights_quant/FakeQuantWithMinMaxVars'):
            tensor_name = tensor_name[24:-30]
            dic['array'] = interpreter.get_tensor(dic['index'])
            dic['array'] = np.moveaxis(dic['array'], 0, -1)
        elif tensor_name.endswith('depthwise/weights_quant/FakeQuantWithMinMaxVars'):
            tensor_name = tensor_name[24:-30]
            dic['array'] = interpreter.get_tensor(dic['index'])
            dic['array'] = np.moveaxis(dic['array'], 0, -1)
        elif tensor_name.endswith('pointwise/weights_quant/FakeQuantWithMinMaxVars'):
            tensor_name = tensor_name[24:-30]
            dic['array'] = interpreter.get_tensor(dic['index'])
            dic['array'] = np.moveaxis(dic['array'], 0, -1)
        elif tensor_name.endswith('Conv2D_Fold_bias'):
            tensor_name = tensor_name[24:-16] + 'bias'
            dic['array'] = interpreter.get_tensor(dic['index'])
        elif tensor_name.endswith('depthwise_Fold_bias'):
            tensor_name = tensor_name[24:-19] + 'bias'
            dic['array'] = interpreter.get_tensor(dic['index'])
        elif tensor_name.endswith('Relu6'):
            tensor_name = tensor_name[24:]
            if_array = 0
        else:
            print('Error: ' + tensor_name)
            if_valid = 0
            if_array = 0

        if quant_type == 'int' and if_array:
            dic['array'] = dic['array'].astype(float)
            dic['array'] = dic['array'] - dic['quantization'][1]

        if if_valid:
            tensor_dic[tensor_name] = dic


    f = open('weights_list.txt','w')
    for key in tensor_dic:
        f.write(key)
        f.write('\n')
    print("a")
        
    return tensor_dic

# load_from_tflite('int')
# test_dic = load_from_pb()




# MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars
# MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_Fold_bias
# MobilenetV1/MobilenetV1/Conv2d_0/Relu6
# MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_Fold_bias
# MobilenetV1/Logits/Conv2d_1c_1x1/weights_quant/FakeQuantWithMinMaxVars
