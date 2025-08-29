import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

income = np.random.randint(20000, 100000, n)
debt = np.random.randint(1000, 40000, n)
payment_history = np.random.randint(0, 2, n)
age = np.random.randint(21, 65, n)
loan_amount = np.random.randint(500, 50000, n)

default = (debt / (income+1) > 0.4).astype(int)
default = np.where(payment_history == 1, np.random.binomial(1, 0.2, n), default)

df = pd.DataFrame({
    "income": income,
    "debt": debt,
    "payment_history": payment_history,
    "age": age,
    "loan_amount": loan_amount,
    "default": default
})

df.to_csv("data/credit_data.csv", index=False)
print("✅ Dataset saved to data/credit_data.csv")