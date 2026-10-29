import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

df = pd.read_csv('total_incidents.csv')
df['Дата'] = pd.to_datetime(df['Дата'])

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.plot(df['Дата'], df['Кол. инцидентов'], color='teal')

# вот тут вся магия — говорим оси показывать только время
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig.autofmt_xdate()

ax.set_title('Инциденты за промежуток времени')
ax.set_xlabel('Дата')
ax.set_ylabel('Кол-во инцидентов')
plt.tight_layout()

plt.show()
plt.savefig('total_incidents.png')
