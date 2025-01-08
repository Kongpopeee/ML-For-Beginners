# นำเข้าไลบรารี NumPy สำหรับการคำนวณ
import numpy as np

# สร้างคลาส Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.1):
        # กำหนดค่าเริ่มต้นสำหรับน้ำหนักและค่าขีดแบ่ง
        self.weights = np.random.randn(2)  # สุ่มค่าน้ำหนักเริ่มต้นสำหรับ 2 features
        self.bias = np.random.randn(1)     # สุ่มค่าขีดแบ่งเริ่มต้น (Threshold)
        self.lr = learning_rate            # กำหนดอัตราการเรียนรู้
    
    def activation(self, x):
        # ฟังก์ชันกระตุ้นแบบขั้นบันได (step function)
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        # คำนวณผลรวมถ่วงน้ำหนักและใช้ฟังก์ชันกระตุ้น
        # สมการ: w₁x₁ + w₂x₂ + θ
        sum_value = np.dot(inputs, self.weights) + self.bias
        return self.activation(sum_value)
    
    def train(self, X, y, epochs=100):
        # ฝึกฝนโมเดลตามจำนวน epochs ที่กำหนด
        for _ in range(epochs):
            # วนลูปผ่านข้อมูลฝึกฝนทีละตัว
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                # ปรับค่าน้ำหนักและค่าโน้มเอียง (bias) ตามความผิดพลาด
                self.weights += self.lr * error * inputs
                self.bias += self.lr * error

# สร้างข้อมูลตัวอย่างสำหรับการฝึกฝน
X = np.array([
    [165, 60],  # คนที่ 1: สูง 165cm หนัก 60kg - ไซส์ X
    [170, 65],  # คนที่ 2: สูง 170cm หนัก 65kg - ไซส์ X
    [175, 80],  # คนที่ 3: สูง 175cm หนัก 80kg - ไซส์ XL
    [180, 85],  # คนที่ 4: สูง 180cm หนัก 85kg - ไซส์ XL
])

# กำหนดเป้าหมาย (labels) สำหรับแต่ละข้อมูล
y = np.array([0, 0, 1, 1])  # 0 = ไซส์ X, 1 = ไซส์ XL

# สร้างและฝึกฝนโมเดล Perceptron
model = Perceptron()
model.train(X, y)

# ทดสอบโมเดลกับข้อมูลใหม่
test_data = np.array([178, 82])  # ทดสอบกับคนสูง 178cm หนัก 82kg
result = model.predict(test_data)
# แสดงผลการทำนาย
print("ขนาดเสื้อที่ทำนาย:", "XL" if result == 1 else "X")
