"""This prints out some statstics on our final results dataset"""
import pandas as pd

df = pd.read_csv("Final_results.csv")
print(df.head())
print(df.columns)
print(len(df))

print(set(df["Modifier"]))

for AI in set(df["AI"]):
    tempdf = df[df["Modifier"].isna()]
    tempdf2 = tempdf[tempdf["AI"] == AI]
    print("For modifier None:\t", AI, "got\t",
        len(tempdf2[tempdf2["Interpretation"] == "Correct"]), "correct \t",
        len(tempdf2[tempdf2["Interpretation"] == "Incorrect"]), "incorrect.")
    for modifier in ["A", "B", "C", "D", "E", "F"]:
        tempdf = df[df["Modifier"] == modifier]
        tempdf2 = tempdf[tempdf["AI"] == AI]
        print("For modifier", modifier, ":\t", AI, "got\t", 
              len(tempdf2[tempdf2["Interpretation"] == "Correct"]), "correct \t",
              len(tempdf2[tempdf2["Interpretation"] == "Incorrect"]), "incorrect.")

for AI in set(df["AI"]):
    tempdf = df[df["Modifier"].isna()]
    tempdf2 = tempdf[tempdf["AI"] == AI]
    print("For modifier None", "\t", AI, "\t", len(tempdf2[tempdf2["Result"] == "Unable"]), "Unables.")
    for modifier in ["A", "B", "C", "D", "E", "F"]:
        tempdf = df[df["Modifier"] == modifier]
        tempdf2 = tempdf[tempdf["AI"] == AI]
        print("For modifier", modifier, "\t", AI, "\t", len(tempdf2[tempdf2["Result"] == "Unable"]), "Unables.")
