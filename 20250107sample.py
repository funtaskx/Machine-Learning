import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read csv file
df = pd.read_csv("biomedical_data.csv")

# replace N/A
df.fillna(df.median(), inplace=True)

# standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.iloc[:, :-1])

# classify training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['label'], test_size=0.2, random_state=42)
