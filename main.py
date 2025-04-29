import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# 🔹 Налаштування сторінки
st.set_page_config(page_title="SolveX - Чисельні методи", layout="centered")

# 📌 Визначаємо змінну x
x = sp.Symbol('x')

# 🔹 Стиль оформлення
st.markdown("""
    <style>
    .block-container { padding: 20px; }
    .stTextInput, .stNumberInput, .stSelectbox { font-size: 18px; padding: 5px; border: 2px solid #007BFF; border-radius: 5px; }
    .stButton>button { font-size: 20px; background-color: #007BFF; color: white; border-radius: 5px; padding: 10px; }
    .error-text { color: red; font-size: 16px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 📌 Заголовок
st.title("🔢 SolveX – Розв’язування рівнянь чисельними методами")
st.subheader("⚡ Введіть рівняння та оберіть метод!")

# 📌 Введення рівняння (з прикладом)
equation_input = st.text_input("✏️ Введіть рівняння:", placeholder="Наприклад: x**3 - x + 2")

# 📌 Вибір методу
method = st.selectbox("🔍 Оберіть метод:", ["Бісекція", "Ньютона", "Хорд"])

# 📌 Межі та точність
a = st.number_input("🔹 Ліва межа (a):", value=-2.0)
b = st.number_input("🔹 Права межа (b):", value=2.0)
tol = st.slider("🎯 Точність (ε):", min_value=0.0001, max_value=0.01, step=0.0001, value=0.001)

# 📌 Кнопка обчислення
if st.button("🚀 Обчислити корінь"):
    try:
        if not equation_input.strip():
            st.markdown('<p class="error-text">❌ Будь ласка, введіть правильне рівняння!</p>', unsafe_allow_html=True)
        else:
            equation = sp.sympify(equation_input)
            func = sp.lambdify(x, equation)

            # 🔹 Метод бісекції
            def bisection_method(func, a, b, tol):
                try:
                    if float(func(a)) * float(func(b)) >= 0:
                        return None
                    while (b - a) / 2 > tol:
                        c = (a + b) / 2
                        if func(c) == 0:
                            return c
                        elif float(func(a)) * float(func(c)) < 0:
                            b = c
                        else:
                            a = c
                    return (a + b) / 2
                except Exception:
                    return None

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
                st.success(f"✅ Знайдений корінь: {root:.6f}")

                fig, ax = plt.subplots(figsize=(6, 4))
                X = np.linspace(a - 1, b + 1, 100)
                Y = func(X)
                ax.plot(X, Y, label="Функція", color="blue")
                ax.axhline(0, color="black", linewidth=1)
                ax.scatter(root, 0, color="red", marker="o", label="Корінь", s=100)
                ax.legend()
                ax.set_title("📊 Графік функції")
                st.pyplot(fig)
            else:
                st.markdown('<p class="error-text">❌ Метод не зміг знайти корінь або задані межі невірні!</p>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<p class="error-text">❌ Введене рівняння має помилку. Перевірте, чи правильно написані символи!</p>', unsafe_allow_html=True)
