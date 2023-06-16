import numpy as np

def bacon(df, max_iter=20):
    for iteracija in range(max_iter):
        stolpci = list(df.columns)
        df_array = np.array(df)
        # ocenimo, ali obstaja konstanta
        sig = np.std(df_array, axis=0)
        # testirajmo se trivialnost
        if min(sig) < 10 ** -12:  # pazimo na skalo?
            break
        # izracunajmo vrstne rede, najdemo korelacije med njimi
        vrstni_redi = np.array(df.rank(axis=0)).T
        korelacije = np.corrcoef(vrstni_redi)  # korelacije[i, j]
        n = korelacije.shape[0]
        korelacije = [(abs(korelacije[i, j]), (i, j), korelacije[i, j] < 0) for i in range(n) for j in range(i + 1, n)]
        korelacije.sort(reverse=True)
        for kakovost, (i, j), je_mnozenje in korelacije:
            if je_mnozenje:
                ime_novega = f"({stolpci[i]}) * ({stolpci[j]})"
                vrednosti_novega = df_array[:, i] * df_array[:, j]
            else:
                ime_novega = f"({stolpci[i]}) / ({stolpci[j]})"
                vrednosti_novega = df_array[:, i] / df_array[:, j]
            if ime_novega not in stolpci:
                df[ime_novega] = vrednosti_novega
                break
    # najdemo "konstanto"
    df_array = np.array(df)
    sig = np.std(df_array, axis=0)
    i = np.argmin(sig)
    const = np.mean(df_array[:, i])
    print(f"{const:.5e} = {df.columns[i]} (napaka: {sig[i]})")


