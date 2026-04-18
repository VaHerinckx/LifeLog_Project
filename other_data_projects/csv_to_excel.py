import pandas as pd

df = pd.read_csv("/Users/valen/code/VaHerinckx/LifeLog_Project/google_snippets_analysis.csv")

df.to_excel("/Users/valen/code/VaHerinckx/LifeLog_Project/google_snippets_analysis.xlsx", index = False)
