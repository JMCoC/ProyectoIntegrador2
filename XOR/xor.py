import numpy as np  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Input

# ------------------------------  
# 1. Datos de entrenamiento (XOR)  
# ------------------------------  
# Entradas (todas las combinaciones posibles)  
X = np.array([  
    [0, 0],  
    [0, 1],  
    [1, 0],  
    [1, 1]  
])  
  
# Salidas esperadas (tabla XOR)  
y = np.array([  
    [0],  
    [1],  
    [1],  
    [0]  
])

# ------------------------------  
# 2. Definir el modelo  
# ------------------------------  
model = Sequential([  
    Input(shape=(2,)),            # 2 entradas (x1, x2)  
    Dense(4, activation="relu"),  # capa oculta con 4 neuronas ReLU  
    Dense(1, activation="sigmoid") # salida con activación Sigmoid  
])

# ------------------------------  
# 3. Compilar el modelo  
# ------------------------------  
model.compile(  
    optimizer="adam",  
    loss="binary_crossentropy",  
    metrics=["accuracy"]  
)
# optimizer: cómo aprende la red (Adam ajusta los pesos inteligentemente).
# loss: qué tan mal lo está haciendo en cada predicción (Binary Crossentropy mide error binario).  
# metrics: cómo medimos el rendimiento (Accuracy muestra el % de aciertos).

# ------------------------------  
# 4. Entrenar el modelo  
# ------------------------------  
history = model.fit(  
    X, y,  
    epochs=5000,       # muchas épocas porque XOR no es lineal  
    verbose=0         # 0 = sin logs, cámbialo a 1 para ver entrenamiento  
)

# ------------------------------  
# 5. Evaluar y predecir  
# ------------------------------  
print("\nEvaluación final:")  
loss, acc = model.evaluate(X, y, verbose=0)  
print(f"Loss: {loss:.3f}, Accuracy: {acc:.3f}")  
  
print("\nPredicciones XOR:")  
for a, b in X:  
    pred = model.predict(np.array([[a, b]]), verbose=0)  
    print(f"{a} XOR {b} = {round(pred.item(), 3)}")