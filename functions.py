import numpy as np

class Loss:
    """كلاس القاعدة لكل دوال الخسارة"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError
        
    def backward(self, y_pred, y_true):
        raise NotImplementedError

class MSE(Loss):
    """
    متوسط مربع الخطأ (Mean Squared Error)
    """
    def forward(self, y_pred, y_true):
        # المعادلة: متوسط (التوقع - الحقيقة) تربيع
        return np.mean((y_pred - y_true)**2)

    def backward(self, y_pred, y_true):
        if y_true.size == 0:
            return 0 # تجنب القسمة على صفر
        return 2 * (y_pred - y_true) / y_true.size