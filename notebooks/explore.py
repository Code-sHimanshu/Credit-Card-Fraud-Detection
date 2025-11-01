import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# adjust path if you placed the file elsewhere
PATH = "./data/creditcard.csv"
df = pd.read_csv(PATH)

# Basic checks
print("shape:", df.shape)
print("\ncolumns:", df.columns.tolist())
print("\nhead:\n", df.head().to_string(index=False))
print("\ninfo:")
print(df.info())
print("\nsummary statistics (numeric):\n", df.describe().T)

# Class distribution (fraud vs legit)
print("\nClass counts:")
print(df["Class"].value_counts())
print("\nClass proportion:")
print(df["Class"].value_counts(normalize=True))


# class balance bar
sns.countplot(x="Class", data=df)
plt.title("Class distribution (0 = legit, 1 = fraud)")
plt.show()

# amount histogram (log scale for better view)
plt.hist(df["Amount"], bins=100)
plt.yscale('log')
plt.title("Transaction Amount (log scale)")
plt.xlabel("Amount")
plt.show()
