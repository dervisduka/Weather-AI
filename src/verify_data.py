import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def verify():
# Lexojmë kolonën e parë (0) si indeks, pa u shqetësuar për emrin e saj
    df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

    print("--- Raporti i Verifikimit ---")
    
    # A ka vrimat kohore?
    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    missing_hours = expected_range.difference(df.index)
    
    if len(missing_hours) == 0:
        print("✅ Integriteti Kohor: OK (Nuk mungon asnjë orë)")
    else:
        print(f"⚠️ Kujdes: Mungojnë {len(missing_hours)} orë në sekuencë!")

    # A ka vlera Null?
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        print("✅ Pastërtia: OK (0 vlera Null)")
    else:
        print(f"❌ Gabim: Ka ende {nulls} vlera boshe!")

    # Kontrolli i Korrelacionit (A kanë sens të dhënat?)
    # Psh: Kur temperatura rritet, lagështia (rhum) duhet të ulet
    correlation = df[['temp', 'rhum', 'pres']].corr()
    print("\nMatrica e Korrelacionit:")
    print(correlation)

    # Vizualizimi i korrelacionit
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Verifikimi i Lidhjes mes Variablave")
    plt.show()

if __name__ == "__main__":
    verify()