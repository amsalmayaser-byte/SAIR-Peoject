import numpy as np

class Optimizer:
    def update(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    # V-- تم التصحيح هنا --V
    def __init__(self, learning_rate=0.01,momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_w = None
        self.v_b = None

def update(self, layer):
        # تهيئة متغيرات السرعة في أول مرة فقط
        if self.v_w is None:
            self.v_w = np.zeros_like(layer.weights)
            self.v_b = np.zeros_like(layer.bias)

        # معادلة تحديث السرعة (الزخم)
        self.v_w = self.momentum * self.v_w - self.lr * layer.weights_gradient
        self.v_b = self.momentum * self.v_b - self.lr * layer.bias_gradient

        # تحديث الأوزان باستخدام السرعة الجديدة
        layer.weights += self.v_w
        layer.bias += self.v_b

class Adam(Optimizer):
    # V-- تم التصحيح هنا --V
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {} # استخدام قاموس لتخزين العزم لكل طبقة
        self.v = {} # استخدام قاموس لتخزين العزم لكل طبقة
        self.t = 0

    def update(self, layer):
        # التأكد من وجود gradients
        if not hasattr(layer, 'weights_gradient') or not hasattr(layer, 'bias_gradient'):
            return

        # استخدام id الطبقة كمفتاح فريد
        layer_id = id(layer)
        
        if layer_id not in self.m:
            self.m[layer_id] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.bias)}
            self.v[layer_id] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.bias)}

        self.t += 1

        # تحديث العزم الأول والثاني للأوزان
        self.m[layer_id]['w'] = self.beta1 * self.m[layer_id]['w'] + (1 - self.beta1) * layer.weights_gradient
        self.v[layer_id]['w'] = self.beta2 * self.v[layer_id]['w'] + (1 - self.beta2) * (layer.weights_gradient ** 2)
        
        # تحديث العزم الأول والثاني للبايس
        self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * layer.bias_gradient
        self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * (layer.bias_gradient ** 2)

        # تصحيح الانحياز
        m_w_hat = self.m[layer_id]['w'] / (1 - self.beta1 ** self.t)
        v_w_hat = self.v[layer_id]['w'] / (1 - self.beta2 ** self.t)
        m_b_hat = self.m[layer_id]['b'] / (1 - self.beta1 ** self.t)
        v_b_hat = self.v[layer_id]['b'] / (1 - self.beta2 ** self.t)

        # تحديث الأوزان والانحياز
        layer.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
