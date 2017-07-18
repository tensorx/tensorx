import tensorflow as tf

def enum_join(tensor):
    """ Joins each element in a row with its index
    [[1,2,3],[4,5,6]] -> [[0,1],[0,2],[0,3],...]
    """
    shape = tf.shape(tensor)
    enum = tf.range(0,shape[0])
    grid = tf.meshgrid(enum)


