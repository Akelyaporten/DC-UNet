# -*- coding: utf-8 -*-
"""DC-Unet.ipynb
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, add, MaxPool2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model


def Dual_Channel(input_img, f1 ,f2 ,f3):

  init = 'he_uniform'
  activation = 'relu'

  C1L1 = Conv2D(f1, (3,3), padding = 'same', kernel_initializer = init, activation = activation)(input_img)
  BN1 = BatchNormalization()(C1L1)
  C1L2 = Conv2D(f2, (3,3), padding = 'same', kernel_initializer = init, activation = activation)(BN1)
  BN2 = BatchNormalization()(C1L2)
  C1L3 = Conv2D(f3, (3,3), padding = 'same', kernel_initializer = init, activation = activation)(BN2)
  BN3 = BatchNormalization()(C1L3)
  concatenation_layer1 = Concatenate(axis = 3)([BN3, BN2, BN1])

  C2L1 = Conv2D(f1, (3,3), padding = 'same', kernel_initializer = init, activation = activation)(input_img)
  BN4 = BatchNormalization()(C2L1)
  C2L2 = Conv2D(f2, (3,3), padding = 'same', kernel_initializer = init, activation = activation)(BN4)
  BN5 = BatchNormalization()(C2L2)
  C2L3 = Conv2D(f3, (3,3), padding = 'same', kernel_initializer = init, activation = activation)(BN5)
  BN6 = BatchNormalization()(C2L3)
  concatenation_layer2 = Concatenate(axis = 3)([BN6, BN5, BN4])

  output = add([concatenation_layer1, concatenation_layer2])

  return output

def Res_Path(x, f):

  init = 'he_uniform'
  activation = 'relu'

  l_1 = Conv2D(f, (3,3), padding = 'same', kernel_initializer = init, activation = activation)(x)
  B_L1 = BatchNormalization()(l_1)
  l_2 = Conv2D(f, (1,1), padding = 'same', kernel_initializer = init, activation = activation)(x)
  B_L2 = BatchNormalization()(l_2)
  l_3 = add([B_L1, B_L2])

  return l_3


def Build_Model():
  input_img = Input(shape = (256, 256, 3))

  #Encoder part
  DC_BLOCK1 = Dual_Channel(input_img, 8, 17, 26)

  RES_PATH1_L1 = Res_Path(DC_BLOCK1, 32)
  RES_PATH1_L2 = Res_Path(RES_PATH1_L1, 32)
  RES_PATH1_L3 = Res_Path(RES_PATH1_L2, 32)
  RES_PATH1_L4 = Res_Path(RES_PATH1_L3, 32)

  P1 = MaxPool2D()(DC_BLOCK1)
  

  DC_BLOCK2 = Dual_Channel(P1, 17, 35, 53)
 
  RES_PATH2_L1 = Res_Path(DC_BLOCK2, 64)
  RES_PATH2_L2 = Res_Path(RES_PATH2_L1, 64)
  RES_PATH2_L3 = Res_Path(RES_PATH2_L2, 64)

  P2 = MaxPool2D()(DC_BLOCK2)

  DC_BLOCK3 = Dual_Channel(P2, 35, 71, 106)

  RES_PATH3_L1 = Res_Path(DC_BLOCK3, 128)
  RES_PATH3_L2 = Res_Path(RES_PATH3_L1, 128)

  P3 = MaxPool2D()(DC_BLOCK3)

  DC_BLOCK4 = Dual_Channel(P3, 71, 142, 213)

  RES_PATH4_L1 = Res_Path(DC_BLOCK4, 256)

  P4 = MaxPool2D()(DC_BLOCK4)
  
  #Bottleneck
  DC_BLOCK5 = Dual_Channel(P4, 142, 284, 427)
  
  #Decoder part
  up1 = Conv2DTranspose(256, kernel_size = 5, strides = 2, padding = 'same')(DC_BLOCK5)
  Res_add1 = Concatenate(axis = 3)([up1, RES_PATH4_L1])
  DC_BLOCK6 = Dual_Channel(Res_add1, 71, 142, 213)

  up2 = Conv2DTranspose(128, kernel_size = 5, strides = 2, padding = 'same')(DC_BLOCK6)
  Res_add2 = Concatenate(axis = 3)([up2, RES_PATH3_L2])
  DC_BLOCK7 = Dual_Channel(Res_add2, 35, 71, 106)

  up3 = Conv2DTranspose(64, kernel_size = 5, strides = 2, padding = 'same')(DC_BLOCK7)
  Res_add3 = Concatenate(axis = 3)([up3, RES_PATH2_L3])
  DC_BLOCK8 = Dual_Channel(Res_add3, 17, 35, 53)

  up4 = Conv2DTranspose(32, kernel_size = 5, strides = 2, padding = 'same')(DC_BLOCK8)
  Res_add4 = Concatenate(axis = 3)([up4, RES_PATH1_L4])
  DC_BLOCK9 = Dual_Channel(Res_add4, 8, 17, 26)

  output = Conv2D(3, (1,1), padding = 'same', kernel_initializer = 'he_uniform', activation = 'tanh')(DC_BLOCK9)

  model = Model(inputs = [input_img], outputs = [output])

  return model



Model = Build_Model()

Model.summary()
