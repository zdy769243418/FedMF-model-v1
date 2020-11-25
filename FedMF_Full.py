import sys
import time
import numpy as np

from shared_parameter import *
from load_data import item_id_list, user_id_list, train_data, test_data


def user_update(single_user_vector, user_rating_list, encrypted_item_vector):
    """

    :param single_user_vector: user1的潜在特征向量
    :param user_rating_list: user1的评分列表
    :param encrypted_item_vector: 从边缘节点下载的加密后的item矩阵V
    :return:
    """

    # 解密加密的矩阵V
    item_vector = np.array([[private_key.decrypt(e) for e in vector] for vector in encrypted_item_vector],
                           dtype=np.float32)
    # 初始化梯度矩阵，梯度的维度是len(item_vector)行，len(single_user_vector)列
    gradient = np.zeros([len(item_vector), len(single_user_vector)], dtype=float)
    for item_id, rate, _ in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (
                -2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = lr * (-2 * error * single_user_vector + 2 * reg_v * item_vector[item_id])

    encrypted_gradient = [[public_key.encrypt(e, precision=1e-5) for e in vector] for vector in gradient]

    return single_user_vector, encrypted_gradient


def loss():
    loss = []
    # User updates
    for i in range(len(user_id_list)):
        for r in range(len(train_data[user_id_list[i]])):
            item_id, rate, _ = train_data[user_id_list[i]][r]
            error = (rate - np.dot(user_vector[i], item_vector[item_id])) ** 2
            loss.append(error)
    return np.mean(loss)


if __name__ == '__main__':

    # Init process
    # user_vector 7*100
    user_vector = np.zeros([len(user_id_list), hidden_dim]) + 0.01

    # item_vector 40*100
    item_vector = np.zeros([len(item_id_list), hidden_dim]) + 0.01

    t = time.time()
    encrypted_item_vector = [[public_key.encrypt(e, precision=1e-5) for e in vector] for vector in item_vector]

    # server加密item矩阵的时间 t1
    t1 = time.time() - t
    print('Item profile encrypt using', t1, 'seconds')
    for iteration in range(max_iteration):

        print('###################')
        print('Iteration', iteration)

        t = time.time()

        cache_size = (sys.getsizeof(encrypted_item_vector[0][0].ciphertext()) +
                      sys.getsizeof(encrypted_item_vector[0][0].exponent)) * \
                     len(encrypted_item_vector) * \
                     len(encrypted_item_vector[0])
        print('Size of Encrypted-item-vector', cache_size / (2 ** 20), 'MB')
        communication_time = cache_size * 8 / (band_width / 4 * 2 ** 30)
        # 打印从server传输item 矩阵到edge的时间t2，假设传输速率为0.25Gb/s
        t2 = communication_time
        print('transform item matrix from server to edge using a %s Gb/s' % 0.25,
              'communication will use %s seconds' % communication_time)

        # 打印从edge传输item 矩阵到user的时间t3,假设传输速率为1Gb/s
        t3 = communication_time/4
        print('transform item matrix from edge to user using a %s Gb/s' % band_width,
              'communication will use %s seconds' % t3)
        # edge0
        # 保存从user0、user1、user2、user3返回的梯度值
        edge0_encrypted_gradient_from_user = []
        # 保存edge0管理的四个user的本地执行时间
        user_time_list = []
        for i in range(4):
            t = time.time()
            user_vector[i], gradient = user_update(user_vector[i], train_data[user_id_list[i]], encrypted_item_vector)
            # 保存用户i解密、本地更新、计算梯度和加密梯度的时间
            user_time_list.append(time.time() - t)
            print('User-%s update using' % i, user_time_list[-1], 'seconds')
            # 将用户i加密的计算的梯度保存在列表0中
            edge0_encrypted_gradient_from_user.append(gradient)

        # edge1
        # 保存从user4、user5、user6 返回的梯度值
        edge1_encrypted_gradient_from_user = []
        # 保存edge1管理的三个user的本地执行时间
        edge1_user_time_list = []
        for i in range(3):
            t = time.time()
            user_vector[i + 4], gradient = user_update(user_vector[i], train_data[user_id_list[i + 4]],
                                                       encrypted_item_vector)
            # 保存用户i本地更新和计算梯度的时间
            user_time_list.append(time.time() - t)
            # 打印出用户i本地更新和计算梯度的时间
            print('User-%s update using' % str(i + 4), user_time_list[-1], 'seconds')
            # 将用户i计算的梯度保存在列表1中
            edge1_encrypted_gradient_from_user.append(gradient)

        t4 = np.mean(user_time_list)
        print('User0-6 Average time', t4)

        # 打印传输的每个用户加密后的计算的梯度矩阵的时间t5 ，这个加密梯度矩阵的传输时间默认所有用户一致 , 传输速率为1Gb/s
        cache_size = (sys.getsizeof(edge0_encrypted_gradient_from_user[0][0][0].ciphertext()) +
                      sys.getsizeof(edge0_encrypted_gradient_from_user[0][0][0].exponent)) * \
                     len(edge0_encrypted_gradient_from_user[0]) * \
                     len(edge0_encrypted_gradient_from_user[0][0])
        print('Size of Encrypted-gradient', cache_size / (2 ** 20), 'MB')
        communication_time = communication_time + cache_size * 8 / (band_width * 2 ** 30)
        t5 = communication_time
        print('transform CGi from user to edge using a %s Gb/s' % band_width,
              'bandwidth, communication will use %s second' % communication_time)
        t = time.time()

        # 打印每个edge聚合的时间t6
        for g in edge0_encrypted_gradient_from_user:  # count:0,1,2,3
            for i in range(len(encrypted_item_vector)):
                for j in range(len(encrypted_item_vector[i])):
                    encrypted_item_vector[i][j] = encrypted_item_vector[i][j] - g[i][j]

        edge0_update_time = (time.time() - t) * (len(user_id_list) / len(user_id_list))
        t6 = edge0_update_time
        print('edge0 update using', t6, 'seconds')

        for g in edge1_encrypted_gradient_from_user:
            for i in range(len(encrypted_item_vector)):
                for j in range(len(encrypted_item_vector[i])):
                    encrypted_item_vector[i][j] = encrypted_item_vector[i][j] - g[i][j]

        edge1_update_time = (time.time() - t) * (len(user_id_list) / len(user_id_list))
        t7 = edge1_update_time
        print('edge1 update using', t7, 'seconds')
        t8 = (t6+t7)/2
        print('edge avg update time', t8, 'seconds')
        print('edge transform 局部聚合后的加密梯度 to server', t5*4, 'seconds')
        # for user computing loss
        item_vector = np.array([[private_key.decrypt(e) for e in vector] for vector in encrypted_item_vector])
        print('loss', loss())

        print('Costing = user update time + edge update time + server time', t4+t5+t8+t5*4, 'seconds')

    prediction = []
    real_label = []

    # testing
    for i in range(len(user_id_list)):
        p = np.dot(user_vector[i:i + 1], np.transpose(item_vector))[0]

        r = test_data[user_id_list[i]]

        real_label.append([e[1] for e in r])
        prediction.append([p[e[0]] for e in r])

    prediction = np.array(prediction, dtype=np.float32)
    real_label = np.array(real_label, dtype=np.float32)

    print('rmse', np.sqrt(np.mean(np.square(real_label - prediction))))
