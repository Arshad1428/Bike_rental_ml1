import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/raw/bike_rental.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour
df["day_type"] = df["workingday"].map({1: "Working Day", 0: "Weekend/Holiday"})

fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Hourly demand: commuter spikes on working days vs weekends
sns.pointplot(data=df, x="hour", y="count", hue="day_type", ax=axes[0], palette="viridis")
axes[0].set_title("Average Hourly Bike Rentals: Working Days vs. Weekends")
axes[0].set_ylabel("Average Rentals")
axes[0].set_xlabel("Hour of the Day")

# Temperature vs. demand
sns.scatterplot(data=df, x="temp", y="count", alpha=0.1, color="blue", ax=axes[1])
axes[1].set_title("Impact of Temperature on Total Rentals")
axes[1].set_ylabel("Total Rentals")
axes[1].set_xlabel("Temperature (°C)")

# Rental distribution by weather condition
sns.boxplot(data=df, x="weather", y="count", hue="weather", ax=axes[2], palette="coolwarm", legend=False)
axes[2].set_title("Rental Distribution by Weather Condition (1: Clear -> 4: Heavy Rain/Snow)")
axes[2].set_ylabel("Total Rentals")
axes[2].set_xlabel("Weather Condition Category")

plt.tight_layout()
plt.savefig("output_images/eda_summary.png", dpi=150)
print("Saved chart to output_images/eda_summary.png")
