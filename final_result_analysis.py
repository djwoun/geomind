"""This prints out some statstics on our final results dataset"""
import pandas as pd
'''
def find_error_row(file_path, error_position):
    with open(file_path, 'rb') as f:
        byte_count = 0
        row_number = 1
        while byte_count < error_position:
            byte = f.read(1)
            if not byte:
                break
            byte_count += 1
            if byte == b'\n':
                row_number += 1
    return row_number
error_byte_position = 51067
file_path = 'Final_results.csv'
print("Error is on row #", find_error_row(file_path, error_byte_position))
'''

df = pd.read_csv("Final_results.csv")
print(df.head())
print(df.columns)
print(len(df))

tempdf=df[df["Result"].isna()]
print("There are", len(tempdf), "rows missing")

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

