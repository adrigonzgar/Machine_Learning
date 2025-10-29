#####################################
#      EX_2_test_models.py          #
#                                   #
#  Prueba los mejores modelos de CV  #
#     contra el set de TEST         #
#####################################

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- Configuración ---
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CV_RESULTS_FILE = os.path.join(RESULTS_DIR, "cv_results_table.csv")

# ---------------------
# Lista para guardar los resultados finales del TEST
test_results = []

print("--- Iniciando Ejercicio 2 (Parte 2): Prueba en Set de Test ---")

# --- 1. Cargar la tabla de resultados de CV ---
try:
    df_cv_results = pd.read_csv(CV_RESULTS_FILE)
except FileNotFoundError:
    print(f"❌ ¡ERROR! No se encontró el archivo: {CV_RESULTS_FILE}")
    print("Asegúrate de haber ejecutado 'EX_2_cross_validation.py' primero.")
    exit()

print(f"✅ Tabla de resultados de CV cargada desde '{CV_RESULTS_FILE}'")

# --- 2. Encontrar el MEJOR 'max_depth' para CADA dataset ---
# Agrupamos por 'dataset' y nos quedamos con la fila (idx)
# que tiene el máximo 'cv_accuracy'
best_models_idx = df_cv_results.groupby('dataset')['cv_accuracy'].idxmax()
best_models_config = df_cv_results.loc[best_models_idx]

print("\n--- 🏆 Los 4 Modelos 'Campeones' (mejor CV Accuracy) son: ---")
print(best_models_config)
print("----------------------------------------------------------\n")

# --- 3. Bucle: Entrenar y Probar cada uno de los 4 "Campeones" ---
for index, config in best_models_config.iterrows():
    
    dataset_name = config['dataset']
    # Convertir 'Sin Límite' (texto) de nuevo a None (para el modelo)
    if config['max_depth'] == 'Sin Límite':
        best_depth = None
    else:
        best_depth = int(config['max_depth'])
        
    print(f"🔬 Probando Campeón: Dataset='{dataset_name}', max_depth={config['max_depth']}")
    
    # --- 4. Cargar los datos de TRAIN y TEST para este dataset ---
    train_file = os.path.join(DATA_DIR, f"data_{dataset_name}_TRAIN.csv")
    test_file = os.path.join(DATA_DIR, f"data_{dataset_name}_TEST.csv")
    
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"  ❌ ¡ERROR! No se encontraron los archivos para '{dataset_name}'")
        continue

    # --- 5. Preparar datos (X, y) ---
    # Quitamos 'map_name' si existe
    if 'map_name' in df_train.columns:
        df_train = df_train.drop(columns=['map_name'])
    if 'map_name' in df_test.columns:
        df_test = df_test.drop(columns=['map_name'])
        
    X_train = df_train.drop(columns=['action'])
    y_train = df_train['action']
    X_test = df_test.drop(columns=['action'])
    y_test = df_test['action']

    # --- 6. Entrenar el modelo "Campeón" ---
    # Lo entrenamos con su 'best_depth' y con TODOS los datos de TRAIN
    model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    model.fit(X_train, y_train)
    print("  > Modelo entrenado con datos de TRAIN.")
    
    # --- 7. Probar el modelo en el set de TEST ---
    y_pred = model.predict(X_test)
    print("  > Modelo probado con datos de TEST.")

    # --- 8. Calcular Métricas de TEST (ML Performance) ---
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"    > Resultados en TEST:")
    print(f"    > Accuracy:   {test_accuracy:.4f}")
    print(f"    > Precision:  {test_precision:.4f}")
    print(f"    > Recall:     {test_recall:.4f}\n")

    # Guardar estos resultados
    test_results.append({
        "dataset": dataset_name,
        "max_depth": config['max_depth'],
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall
    })

print("=======================================================")
print("✅ ¡Pruebas en Test completadas!")
print("=======================================================")

# --- 9. Guardar la tabla de resultados de TEST ---
if test_results:
    df_test_results = pd.DataFrame(test_results)
    df_test_results = df_test_results.sort_values(by="test_accuracy", ascending=False)
    
    results_file = os.path.join(RESULTS_DIR, "test_results_table.csv")
    df_test_results.to_csv(results_file, index=False)
    
    print(f"🏆 Tabla de resultados de TEST guardada en: {results_file}")
    print("\n--- MEJORES MODELOS (según Test Accuracy) ---")
    print(df_test_results.head())
else:
    print("⚠️ No se generaron resultados de test.")

print("\n--- Fin del Ejercicio 2 (Parte 2) ---")
