import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings

# Suprimir warnings
warnings.filterwarnings("ignore")

print("--- Iniciando Prueba de Modelos en Set de Test (EJ 2.3) ---")

# --- Configuración ---
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CV_RESULTS_FILE = os.path.join(RESULTS_DIR, "cv_results_table.csv")

datasets = ["original", "new_features", "kbest", "rfe"]
results = []

# --- 1. Leer la tabla de CV para encontrar a los "campeones" ---
try:
    # --- ¡ARREGLO CLAVE! Leer con COMA (por defecto) ---
    df_cv = pd.read_csv(CV_RESULTS_FILE)
except FileNotFoundError:
    print(f"❌ ¡ERROR! No se encuentra el archivo {CV_RESULTS_FILE}")
    print("Por favor, ejecuta 'EX_2_cross_validation.py' primero.")
    exit()

print(f"✅ Leída la tabla de resultados de {CV_RESULTS_FILE}")

# Asegurarse de que las columnas numéricas son leídas como números
df_cv['mean_accuracy'] = pd.to_numeric(df_cv['mean_accuracy'])

champion_models = {}
for name in datasets:
    df_dataset = df_cv[df_cv['dataset'] == name]
    if df_dataset.empty:
        print(f"⚠️ No hay datos de CV para {name}. Saltando.")
        continue
    best_row = df_dataset.loc[df_dataset['mean_accuracy'].idxmax()]
    champion_models[name] = best_row['max_depth']

print("\n🏆 'Campeones' elegidos (mejor max_depth de cada dataset):")
print(champion_models)

# --- 2. Entrenar y Probar los 4 campeones ---
for name, depth in champion_models.items():
    print(f"\n--- Probando Campeón: '{name}' (depth={depth}) ---")
    
    try:
        # --- ¡ARREGLO CLAVE! Leer con COMA (por defecto) ---
        df_train = pd.read_csv(os.path.join(DATA_DIR, f"data_{name}_TRAIN.csv"))
        df_test = pd.read_csv(os.path.join(DATA_DIR, f"data_{name}_TEST.csv"))
    except FileNotFoundError:
        print(f"⚠️ ¡Aviso! No se encontraron archivos para '{name}'. Saltando...")
        continue
    except pd.errors.EmptyDataError:
        print(f"⚠️ ¡Aviso! Archivos de datos para '{name}' están vacíos.")
        continue

    # Preparar datos
    if 'map_name' in df_train.columns:
        df_train = df_train.drop(columns=['map_name'])
    if 'map_name' in df_test.columns:
        df_test = df_test.drop(columns=['map_name'])
    
    if df_train.empty or df_test.empty:
        print(f"⚠️ ¡Aviso! Datos vacíos para '{name}' después de procesar.")
        continue

    X_train = df_train.drop(columns=['action'])
    y_train = df_train['action']
    X_test = df_test.drop(columns=['action'])
    y_test = df_test['action']

    # Crear y Entrenar el modelo
    model_depth = None if (str(depth) == "None" or pd.isna(depth)) else int(float(depth))
    
    model = DecisionTreeClassifier(max_depth=model_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predecir en TEST
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy en Test: {accuracy:.4f}")
    print(f"  Precision en Test: {precision:.4f}")
    print(f"  Recall en Test: {recall:.4f}")

    results.append({
        "dataset": name,
        "max_depth": depth,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

# --- 3. Guardar la tabla final de resultados de Test ---
if not results:
    print("\n❌ ¡ERROR! No se generaron resultados de test.")
else:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="test_accuracy", ascending=False)
    output_path = os.path.join(RESULTS_DIR, "test_results_table.csv")

    # --- ¡ARREGLO CLAVE! Guardar con COMA (por defecto) ---
    df_results.to_csv(output_path, index=False)

    print("\n--- ¡Prueba completada! ---")
    print(f"✅ Tabla de resultados de Test guardada en: {output_path}")
    print("\nContenido de la tabla:")
    print(df_results)

