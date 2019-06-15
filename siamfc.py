from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import math
import numpy as np
import cv2
from collections import namedtuple

class SiamFC():

    def __init__(self):
        super(SiamFC, self).__init__()

	def conv_bn_layer(self, input, num_filters, filter_size, stride=1, groups=1, act=None, bn=True, bias_attr=False):
	    conv = fluid.layers.conv2d(
		    input=input,
		    num_filters=num_filters,
		    filter_size=filter_size,
		    stride=stride,
		    padding=(filter_size - 1) // 2,
		    groups=groups,
		    act=None,
		    bias_attr=bias_attr,
		    param_attr=ParamAttr(initializer=MSRA()))
	    if bn == True:
		  conv = fluid.layers.batch_norm(input=conv, act=act, momentum=0.05)
	    return conv

    def alexnet(self, input):
        conv1 = self.conv_bn_layer(input=input, 
			num_filters=96, 
			filter_size=11, 
			stride=2, 
			act='relu')
		conv1 = self.fluid.layers.pool2d(
			input=conv1, 
			pool_size=3, 
			pool_stride=2, 
			pool_type='max')
		conv2 = self.conv_bn_layer(input=conv1, 
			num_filters=256, 
			filter_size=5, 
			stride=1, 
			groups=2,
			act='relu')
		conv2 = self.fluid.layers.pool2d(
			input=conv2, 
			pool_size=3, 
			pool_stride=2, 
			pool_type='max')
        conv3 = self.conv_bn_layer(input=conv2, 
			num_filters=384, 
			filter_size=3, 
			stride=1, 
			act='relu')
		conv4 = self.conv_bn_layer(input=conv3, 
			num_filters=384, 
			filter_size=3, 
			stride=1, 
			groups=2,
			act='relu')
		conv5 = self.conv_bn_layer(input=conv4, 
			num_filters=256, 
			filter_size=3, 
			stride=1, 
			groups=2,
			act='relu')
		out = conv5
        return out

	def net(self, z, x):
		z = self.alexnet(z)
		x = self.alexnet(x)

class TrackerSiamFc():

   def __init__(self, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__(
            name='SiamFC', is_deterministic=True)