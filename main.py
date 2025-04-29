import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# 🔹 Налаштування сторінки
st.set_page_config(page_title="SolveX - Чисельні методи", layout="centered")

# 📌 Визначаємо змінну x
x = sp.Symbol('x')

# 📌 Введення рівняння
st.title("SolveX - Розв’язування рівнянь чисельними методами")
equation_input = st.text_input("Введіть рівняння (наприклад, x**3 - x - 2)")

# 📌 Вибір методу
method = st.selectbox("Оберіть метод:", ["Бісекція", "Ньютона", "Хорд"])

# 📌 Межі та точність
a = st.number_input("Ліва межа (a)", value=-2.0)
b = st.number_input("Права межа (b)", value=2.0)
tol = st.slider("Точність (ε)", min_value=0.0001, max_value=0.01, step=0.0001, value=0.001)

# 📌 Перевірка введення
if equation_input:
    equation = sp.sympify(equation_input)
    func = sp.lambdify(x, equation)

    # 🔹 Метод бісекції
    def bisection_method(func, a, b, tol):
        if func(a) * func(b) >= 0:
            return None
        while (b - a) / 2 > tol:
            c = (a + b) / 2
            if func(c) == 0:
                return c
            elif func(a) * func(c) < 0:
                b = c
            else:
                a = c
        return (a + b) / 2

    # 🔹 Метод Ньютона
    def newton_method(func, x0, tol):
        df = sp.diff(equation, x)
        df_func = sp.lambdify(x, df)
        x_curr = x0
        for _ in range(100):
            x_next = x_curr - func(x_curr) / df_func(x_curr)
            if abs(x_next - x_curr) < tol:
                return x_next
            x_curr = x_next
        return None

    # 🔹 Метод хорд
    def secant_method(func, a, b, tol):
        x0, x1 = a, b
        for _ in range(100):
            x_next = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
            if abs(x_next - x1) < tol:
                return x_next
            x0, x1 = x1, x_next
        return None

    # 📌 Розрахунок кореня
    root = None
    if method == "Бісекція":
        root = bisection_method(func, a, b, tol)
    elif method == "Ньютона":
        root = newton_method(func, (a + b) / 2, tol)
    elif method == "Хорд":
        root = secant_method(func, a, b, tol)

    # 📌 Візуалізація
    if root is not None:
        st.success(f"Знайдений корінь: {root:.6f}")

        fig, ax = plt.subplots(figsize=(6, 4))
        X = np.linspace(a - 1, b + 1, 100)
        Y = func(X)
        ax.plot(X, Y, label="Функція", color="blue")
        ax.axhline(0, color="black", linewidth=1)
        ax.scatter(root, 0, color="red", marker="o", label="Корінь")
        ax.legend()
        ax.set_title("Графік функції")
        st.pyplot(fig)
    else:
        st.error("Метод не зміг знайти корінь або задані межі невірні.")
