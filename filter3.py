import pandas as pd

df = pd.read_csv("final_filtered.csv", parse_dates=["lastReportedAt"])

df["window"] = df["lastReportedAt"].dt.floor("30s")
result = df.groupby("window").size().reset_index(name="events_count")

result.to_csv('total_incidents.csv', index=False)