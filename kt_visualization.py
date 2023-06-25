import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

SKILL_ID = "it-network-plan"

# Load CSV
df = pd.read_csv('abzuege/tracing.csv')

# Convert "Mastery" to floating point numbers
df['Mastery'] = df['Mastery'].str.replace(',', '.').astype(float)

# Convert "Date" to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S')

# Sort the dataframe by 'Date'
df = df.sort_values('Date')

# Select a user
user_id = 'afe3ee14-2243-4133-83b5-4f7ff6aafd78'

# Filter the dataframe for the selected user
df_user = df[df['UserId'] == user_id]

# Plot mastery over time
fig, ax = plt.subplots(figsize=(10, 6))
for skill in df_user['SkillId'].unique():
    df_skill = df_user[df_user['SkillId'] == skill]
    ax.plot(df_skill['Date'], df_skill['Mastery'], label=skill)

# Set the date format
date_format = mdates.DateFormatter('%d.%m.%Y')
ax.xaxis.set_major_formatter(date_format)

# Set the date interval
ax.xaxis.set_major_locator(mdates.DayLocator())

plt.xlabel('Date')
plt.ylabel('Mastery')
plt.legend()
plt.title(f'Mastery Progress for User: {user_id}')
plt.show()
