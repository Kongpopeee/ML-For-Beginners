import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        # เก็บค่า weights และ biases สำหรับแต่ละชั้น
        self.weights = []
        self.biases = []
        
        # สร้าง weights และ biases สำหรับแต่ละชั้น
        # weights: สุ่มค่าเริ่มต้นด้วย normal distribution
        # biases: เริ่มต้นด้วยค่าศูนย์
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1])  # สร้าง weight matrix
            b = np.zeros((1, layer_sizes[i+1]))  # สร้าง bias vector
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        # ฟังก์ชันกระตุ้นแบบ sigmoid: f(x) = 1/(1 + e^(-x))
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # อนุพันธ์ของฟังก์ชัน sigmoid: f'(x) = f(x)(1 - f(x))
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, x):
        # เก็บค่า activation ของแต่ละชั้น โดยเริ่มจาก input
        self.activations = [x]
        # เก็บค่า z (ผลรวมถ่วงน้ำหนักก่อนผ่านฟังก์ชันกระตุ้น)
        self.z_values = []
        
        # Forward propagation: คำนวณไปทีละชั้นจาก input ไปยัง output
        for i in range(len(self.weights)):
            # คำนวณผลรวมถ่วงน้ำหนัก: z = wx + b
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            # ผ่านฟังก์ชันกระตุ้น sigmoid
            activation = self.sigmoid(z)
            self.activations.append(activation)
            
        return self.activations[-1]  # ส่งคืนค่า output layer
    
    def backward(self, x, y, learning_rate=0.1):
        # m คือจำนวนตัวอย่างข้อมูล
        m = x.shape[0]
        # คำนวณค่าความผิดพลาดที่ output layer
        delta = self.activations[-1] - y
        
        # Backward propagation: ปรับ weights และ biases จาก output ไป input
        for i in range(len(self.weights) - 1, -1, -1):
            # คำนวณ gradient ของ weights และ biases
            dW = np.dot(self.activations[i].T, delta) / m  # gradient ของ weights
            db = np.sum(delta, axis=0, keepdims=True) / m  # gradient ของ biases
            
            # คำนวณ delta สำหรับชั้นถัดไป (ยกเว้นชั้น input)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i-1])
            
            # ปรับค่า weights และ biases ตาม gradient descent
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db


# สร้าง MLP ที่มี 2 inputs, 3 hidden neurons, และ 1 output
mlp = MLP([2, 3, 1])

# ข้อมูลตัวอย่าง XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# ฝึกฝนโมเดล
for epoch in range(10000):
    # Forward pass
    output = mlp.forward(X)
    
    # Backward pass
    mlp.backward(X, y)
    
    if epoch % 1000 == 0:
        loss = np.mean(np.square(output - y))
        print(f'Epoch {epoch}, Loss: {loss}')

# ทดสอบโมเดล
predictions = mlp.forward(X)
print("\nPredictions:")
print(predictions)
