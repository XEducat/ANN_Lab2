# Реалізація одношарового персептрону для XOR
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):  # ← виправлено
        self.weights = np.zeros(input_size + 1)  # +1 для bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation >= 0 else 0

    def train(self, training_inputs, labels, epochs=20):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# === 1. Дані для функції XOR ===
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 1, 1, 0])

# === 2. Тренування звичайного персептрону ===
perceptron_linear = Perceptron(input_size=2, learning_rate=0.1)
perceptron_linear.train(training_inputs, labels, epochs=20)

# Перевірка результатів
print("Результати звичайного одношарового персептрону (невдача):")
for inputs in training_inputs:
    print(f"{inputs} => {perceptron_linear.predict(inputs)}")

# === Візуалізація невдачі ===
plt.figure(figsize=(6, 5))
plt.title("Одношаровий персептрон (невдача на XOR)")
plt.scatter(training_inputs[:, 0], training_inputs[:, 1],
            c=labels, cmap='bwr', edgecolors='k', s=100)
plt.xlabel("x1")
plt.ylabel("x2")
plt.text(0.1, 1.1, "Модель не може провести межу між класами",
         fontsize=9, color='gray')
plt.show()

# === 3. Додавання проміжної ознаки (x1*x2) ===
extended_inputs = np.array([[x1, x2, x1 * x2] for x1, x2 in training_inputs])

# Новий персептрон з трьома входами
perceptron_nonlinear = Perceptron(input_size=3, learning_rate=0.1)
perceptron_nonlinear.train(extended_inputs, labels, epochs=20)

# Перевірка результатів
print("\nРезультати модифікованого персептрону (успіх):")
for inputs in extended_inputs:
    print(f"{inputs} => {perceptron_nonlinear.predict(inputs)}")

# === Візуалізація успіху ===
x1_range = np.linspace(-0.5, 1.5, 100)
x2_range = np.linspace(-0.5, 1.5, 100)
xx, yy = np.meshgrid(x1_range, x2_range)
zz = np.array([perceptron_nonlinear.predict([x, y, x * y])
               for x, y in zip(xx.ravel(), yy.ravel())])
zz = zz.reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, alpha=0.3, cmap='bwr')
plt.scatter(training_inputs[:, 0], training_inputs[:, 1],
            c=labels, cmap='bwr', edgecolors='k', s=100)
plt.title("Модифікований персептрон для XOR (успішна класифікація)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()