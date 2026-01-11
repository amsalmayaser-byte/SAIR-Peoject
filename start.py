import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- استيراد الكلاسات من مكتبتك ---
from nn_library.core.layer import Dense
from nn_library.core.network import NeuralNetwork
from nn_library.losses.functions import MSE
from nn_library.optimizers.optimizers import Adam

print("--- Script Started: Breast Cancer Classification (with Batch Training) ---")

# --- 1. تحميل وإعداد البيانات ---
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_reshaped = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reshaped, test_size=0.2, random_state=42)
print("Data loaded and prepared.")

# --- 2. بناء الشبكة ---
n_features = X_train.shape[1]
model = NeuralNetwork()
model.add(Dense(input_size=n_features, output_size=16, activation='relu',dropout_rate=0.2,l2_lambda=0.01))
model.add(Dense(input_size=16, output_size=8, activation='relu',dropout_rate=0.2,l2_lambda=0.01))
model.add(Dense(input_size=8, output_size=1, activation='sigmoid',l2_lambda=0.01))
print("Model built with Dropout and L2 Regularization.")

# --- 3. تحديد دالة الخسارة والمُحسِّن ---
model.set_loss(MSE())
optimizer = Adam(learning_rate=0.001)
print("Loss and optimizer set (Using Adam).")

# --- 4. تدريب النموذج ---
print("Starting training...")
# الآن سيتم استخدام Batch Training تلقائياً مع حجم دفعة 32
loss_history = model.train(X_train, y_train, epochs=100, optimizer=optimizer)
print("Training finished.")

# --- 5. تقييم النموذج (حساب الدقة) ---
print("\n--- Evaluating Model ---")
# للحصول على الدقة، يجب أن نمرر البيانات دفعة واحدة عبر predict
predictions = model.predict(X_test)
# .round() سيحول الاحتمالات (e.g., 0.98) إلى 0 أو 1
predicted_classes = predictions.round()
accuracy = np.mean(predicted_classes == y_test) * 100
print(f"Accuracy on test data: {accuracy:.2f}%")

# --- 6. تصور النتائج (رسم منحنى الخسارة) ---
print("Plotting loss curve...")
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Model Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.savefig("loss_curve_batch.png") # حفظ الرسمة باسم جديد
plt.show()
print("Loss curve saved as loss_curve_batch.png")
