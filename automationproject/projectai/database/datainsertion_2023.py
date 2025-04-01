import pandas as pd
from projectai.LLMS.supabase_client import supabase_client


file_path="/home/sridhar/AI_Spring_2025/1. students/s.sunke/automationproject/projectai/data/BLACK ENTERTAINMENT TV_SmartGridView_2023.xlsx"

df=pd.read_excel(file_path)


# Convert columns to correct data types
df["Start Date"] = pd.to_datetime(df["Start Date"]).dt.strftime('%Y-%m-%d')
df["End Date"] = pd.to_datetime(df["End Date"]).dt.strftime('%Y-%m-%d')

# Convert Start Time and End Time to string (HH:MM:SS)
df["Start Time"] = pd.to_datetime(df["Start Time"], format='%I:%M%p').dt.strftime('%H:%M:%S')
df["End Time"] = pd.to_datetime(df["End Time"], format='%I:%M%p').dt.strftime('%H:%M:%S')


# Convert float columns (ratings, coverage, etc.)
float_columns = [
    "National Rating", "National Rating PW", "National Rating P4W", "National Rating P8W", 
    "National Rating YOY", "Coverage Rating", "Coverage Rating PW", "Coverage Rating P4W", 
    "Coverage Rating P8W", "Coverage Rating YOY", "Network Daypart Avg P52W", "National Share", 
    "National Share PW", "National Share P4W", "National Share P8W", "National Share YOY", 
    "Coverage Share", "Coverage Share PW", "Coverage Share P4W", "Coverage Share P8W", 
    "Coverage Share YOY", "Reach", "Reach P4W", "Reach YOY", "TSV", "TSV PW", "TSV P4W", 
    "TSV P8W", "TSV YOY", "Retention"
]
df[float_columns] = df[float_columns].apply(pd.to_numeric, errors="coerce")

# Convert int columns (AA, Reach, etc.)
int_columns = [
    "National AA (000s)", "National AA (000s) PW", "National AA (000s) P8W", "Coverage AA (000s)",
    "Coverage AA (000s) PW", "Coverage AA (000s) P8W", "Reach PW", "Reach P8W"
]
df[int_columns] = df[int_columns].apply(pd.to_numeric, errors="coerce").astype("Int64")

# Convert object columns (Feed Pattern Indicator, Premiere?, Movie Indicator, Special Indicator)
df["Feed Pattern Indicator"] = df["Feed Pattern Indicator"].astype("category")  # If categorical
df["Premiere?"] = df["Premiere?"].map({"Y": True, "N": False})  # Convert Yes/No to boolean
df["Movie Indicator"] = df["Movie Indicator"].map({"Y": True, "N": False})  # Convert Yes/No to boolean
df["Special Indicator"] = df["Special Indicator"].map({"Y": True, "N": False})  # Convert Yes/No to boolean

# Handle NaN (replace with None for Supabase compatibility)
df = df.astype(object).where(pd.notnull(df), None)

# Convert DataFrame to JSON-like format for Supabase
data = df.to_dict(orient="records")

# Your batch insert logic here (e.g., `batch_insert(data, BATCH_SIZE)`)
BATCH_SIZE = 500
TABLE_NAME="rsg_data_2023"

def batch_insert(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        response = supabase_client.table(TABLE_NAME).insert(batch).execute()
        print(f"Inserted {len(batch)} records: {response}")

batch_insert(data,BATCH_SIZE)