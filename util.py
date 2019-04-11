import tensorflow as tf
import numpy as np


def tensor_expand(tensor_Input,Num):
    '''
    张量自我复制扩展，将Num个tensor_Input串联起来，生成新的张量，
    新的张量的shape=[tensor_Input.shape,Num]
    :param tensor_Input:
    :param Num:
    :return:
    '''
    tensor_Input = tf.expand_dims(tensor_Input,axis=0)
    tensor_Output = tensor_Input
    for i in range(Num-1):
        tensor_Output= tf.concat([tensor_Output,tensor_Input],axis=0)
    return tensor_Output


def get_one_hot_matrix(height, width, position):
    '''
    生成一个 one_hot矩阵，shape=【height*width】，在position处的元素为1，其余元素为0
    :param height:
    :param width:
    :param position: 格式为【h_Index,w_Index】,h_Index,w_Index为int格式
    :return:
    '''
    col_one_position = position[0]
    row_one_position = position[1]
    rows_num = height
    cols_num = width

    single_row_one_hot = tf.one_hot(row_one_position, width, dtype=tf.float32)
    single_col_one_hot = tf.one_hot(col_one_position, height, dtype=tf.float32)

    one_hot_rows = tensor_expand(single_row_one_hot, rows_num)
    one_hot_cols = tensor_expand(single_col_one_hot, cols_num)
    one_hot_cols = tf.transpose(one_hot_cols)

    one_hot_matrix = one_hot_rows * one_hot_cols
    return one_hot_matrix
