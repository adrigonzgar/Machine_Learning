#############################
#   feature_selection.py    #
#############################
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

print("Iniciando Selección de Características...")

# --- Configuración ---
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed" # Guardaremos todo en la misma carpeta
K_FEATURES = 3 # Número de features que queremos seleccionar (ej: 3)

# --- Cargar los datasets (solo los de "new features") ---
try:
    df_train = pd.read_csv(os.path.join(INPUT_DIR, "data_new_features_TRAIN.csv"))
    df_test = pd.read_csv(os.path.join(INPUT_DIR, "data_new_features_TEST.csv"))
except FileNotFoundError:
    print("❌ ¡ERROR! Asegúrate de tener los archivos 'data_new_features_TRAIN.csv' y 'TEST.csv' en 'data/processed/'")
    exit()

print("✅ Archivos 'data_new_features' cargados.")

# --- Preparar los datos ---
# Quitamos 'map_name' porque no es una feature para el modelo
if 'map_name' in df_train.columns:
    df_train = df_train.drop(columns=['map_name'])
if 'map_name' in df_test.columns:
    df_test = df_test.drop(columns=['map_name'])

# Separamos en X (features) e y (target, o sea 'action')
X_train = df_train.drop(columns=['action'])
y_train = df_train['action']

X_test = df_test.drop(columns=['action'])
y_test = df_test['action']

# Guardamos los nombres de las features originales
feature_names = X_train.columns

# ===================================================================
#  Método 1: SelectKBest (Selecciona las K Mejores)
# ===================================================================
print(f"\n--- Aplicando Método 1: SelectKBest (k={K_FEATURES}) ---")

# 1. Creamos el selector. Usamos 'f_classif' porque es bueno para
#    clasificación y funciona bien con los números negativos (-1) que tenemos.
selector_kbest = SelectKBest(score_func=f_classif, k=K_FEATURES)

# 2. "Entrenamos" el selector (solo con datos de TRAIN)
#    Aprende cuáles son las K mejores features
selector_kbest.fit(X_train, y_train)

# 3. Obtenemos los nombres de las features elegidas (¡importante para tu informe!)
kbest_indices = selector_kbest.get_support(indices=True)
kbest_features = feature_names[kbest_indices].tolist()
print(f"Features seleccionadas por KBest: {kbest_features}")

# 4. "Transformamos" (filtramos) nuestros datasets
X_train_kbest = selector_kbest.transform(X_train)
X_test_kbest = selector_kbest.transform(X_test)

# 5. Creamos los nuevos DataFrames y les añadimos 'action'
df_kbest_train = pd.DataFrame(X_train_kbest, columns=kbest_features)
df_kbest_train['action'] = y_train.values

df_kbest_test = pd.DataFrame(X_test_kbest, columns=kbest_features)
df_kbest_test['action'] = y_test.values

# 6. Guardamos los archivos (Este es el Dataset 3)
df_kbest_train.to_csv(os.path.join(OUTPUT_DIR, "data_kbest_TRAIN.csv"), index=False)
df_kbest_test.to_csv(os.path.join(OUTPUT_DIR, "data_kbest_TEST.csv"), index=False)
print("✅ Dataset 'kbest' (Dataset 3) guardado.")

# ===================================================================
#  Método 2: RFE (Recursive Feature Elimination)
# ===================================================================
print(f"\n--- Aplicando Método 2: RFE (n={K_FEATURES}) ---")

# 1. RFE necesita un modelo (estimador) para funcionar.
#    Como en el Ej. 2 usaremos Árboles, usamos un Árbol aquí.
estimator = DecisionTreeClassifier(random_state=42)

# 2. Creamos el selector RFE
selector_rfe = RFE(estimator, n_features_to_select=K_FEATURES, step=1)

# 3. "Entrenamos" el selector RFE (esto tarda un poquito más)
selector_rfe.fit(X_train, y_train)

# 4. Obtenemos los nombres de las features elegidas (¡para tu informe!)
rfe_indices = selector_rfe.get_support(indices=True)
rfe_features = feature_names[rfe_indices].tolist()
print(f"Features seleccionadas por RFE: {rfe_features}")

# 5. "Transformamos" (filtramos) nuestros datasets
X_train_rfe = selector_rfe.transform(X_train)
X_test_rfe = selector_rfe.transform(X_test)

# 6. Creamos los nuevos DataFrames y les añadimos 'action'
df_rfe_train = pd.DataFrame(X_train_rfe, columns=rfe_features)
df_rfe_train['action'] = y_train.values

df_rfe_test = pd.DataFrame(X_test_rfe, columns=rfe_features)
df_rfe_test['action'] = y_test.values

# 7. Guardamos los archivos (Este es el Dataset 4)
df_rfe_train.to_csv(os.path.join(OUTPUT_DIR, "data_rfe_TRAIN.csv"), index=False)
df_rfe_test.to_csv(os.path.join(OUTPUT_DIR, "data_rfe_TEST.csv"), index=False)
print("✅ Dataset 'RFE' (Dataset 4) guardado.")

print("\n--- ¡Selección de características completada! ---")