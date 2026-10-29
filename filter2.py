"""
Группировка событий в инциденты по временному окну.

Правило: если в скользящем окне N секунд встречается >= min_ips
РАЗНЫХ ip-адресов — это один инцидент. Все события внутри окна
получают один incident_id. Окна не пересекаются: после закрытия
инцидента поиск следующего окна начинается со следующего
"непроглоченного" события.
"""

import pandas as pd
from datetime import timedelta


def group_incidents(
    df: pd.DataFrame,
    window_seconds: int = 30,
    min_ips: int = 1,
    ip_col: str = "ipAddress",
    time_col: str = "lastReportedAt",
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    incident_ids = [None] * n
    incident_counter = 0

    i = 0
    while i < n:
        window_end = df.loc[i, time_col] + timedelta(seconds=window_seconds)

        j = i
        ips_in_window = set()
        while j < n and df.loc[j, time_col] <= window_end:
            ips_in_window.add(df.loc[j, ip_col])
            j += 1

        if len(ips_in_window) >= min_ips:
            incident_counter += 1
            incident_label = f"incident_{incident_counter}"
            for idx in range(i, j):
                incident_ids[idx] = incident_label
            i = j  # окно закрыто, едем дальше с этой точки
        else:
            i += 1  # окно не набрало кворум, сдвигаемся на 1 событие

    df["incident_id"] = incident_ids
    return df


if __name__ == "__main__":
    data = 'new_filtered.csv'
    df = pd.read_csv(data)
    result = group_incidents(df)
    result.to_csv('final_filtered.csv', index=False)