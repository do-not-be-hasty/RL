from mrunner.helpers.client_helper import get_configuration

from learn import learn_Sokoban_HER


# import tensorflow as tf
# slim = tf.contrib.slim
#
# def convnet_mnist(output=1):
#     def network_fn(state):
#         net = tf.cast(state, tf.float32)
#
#         for _ in range(5):
#             net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME')
#
#         # net = slim.max_pool2d(net, [2, 2], scope='pool2')
#         # net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME')
#         # net = slim.max_pool2d(net, [2, 2])
#
#         net = slim.flatten(net)
#         net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu)
#         value = slim.fully_connected(net, output, activation_fn=None)
#
#         return value
#
#     return network_fn


def main():
    params = get_configuration(print_diagnostics=True, with_neptune=True)

    # learn_BitFlipper_HER()
    # learn_Maze_HER()
    # learn_Maze_MTR()
    # learn_Sokoban_DQN()
    learn_Sokoban_HER()


if __name__ == '__main__':
    main()
