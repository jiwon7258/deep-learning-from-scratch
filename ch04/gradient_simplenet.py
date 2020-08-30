# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])  # 입력 데이터
t = np.array([0, 0, 1])  # 정답 레이블

net = simpleNet()

f = lambda w: net.loss(x, t)  # 손실 함수값을 반환
dW = numerical_gradient(f, net.W)  # 신경망의 기울기를 반환

print(dW)
