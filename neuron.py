import numpy as np
# استيراد الدوال التي جهزناها سابقاً في الملفات الأخرى
from nn_library.activations.functions import Sigmoid, ReLU, Tanh

class Neuron:
    def __init__(self, n_inputs, activation='sigmoid'):
        self.w = np.random.randn(n_inputs) * 0.01 
        self.b = 0.0
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'relu':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            self.activation = None

    def forward(self, x):

        self.x = x # نحفظ المدخلات لأننا سنحتاجها في الباكورد
        self.z = np.dot(self.w, x) + self.b
        
        # إذا كان هناك دالة تنشيط، نمرر النتيجة من خلالها
        if self.activation:
            return self.activation.forward(self.z)
        return self.z

    def backward(self, dout, learning_rate=0.01):
        if self.activation:
            dz = self.activation.backward(dout)
        else:
            dz = dout
        dw = dz * self.x
        db = dz
        self.w -= learning_rate * dw
        self.b -= learning_rate * db
    
        return dw, db