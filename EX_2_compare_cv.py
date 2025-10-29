#####################################
#    EX_2_compare_cv.py           #
#                                   #
#  Compara la nota de CV (fiable)   #
# vs. la nota de un Split (suerte)  #
#####################################

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuración ---
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CV_RESULTS_FILE = os.path.join(RESULTS_DIR, "cv_results_table.csv")

# ---------------------
# Lista para guardar los resultados de la comparación
comparison_results = []

print("--- Iniciando Ejercicio 2 (Parte 3): Comparación CV vs. Split Simple ---")

# --- 1. Cargar la tabla de resultados de CV ---
try:
    df_cv_results = pd.read_csv(CV_RESULTS_FILE)
except FileNotFoundError:
    print(f"❌ ¡ERROR! No se encontró el archivo: {CV_RESULTS_FILE}")
    print("Asegúrate de haber ejecutado 'EX_2_cross_validation.py' primero.")
    exit()

# --- 2. Encontrar el "Top 3" MEJORES modelos de todos ---
# Ordenamos por 'cv_accuracy' y cogemos los 3 primeros
top_3_models = df_cv_results.sort_values(by="cv_accuracy", ascending=False).head(3)

print("\n--- 🏆 El Top 3 de modelos (según CV) es: ---")
print(top_3_models)
print("--------------------------------------------------\n")

# --- 3. Bucle: Probar cada uno de los 3 "Top" ---
for index, config in top_3_models.iterrows():
    
    dataset_name = config['dataset']
    # Convertir 'Sin Límite' (texto) de nuevo a None (para el modelo)
    if config['max_depth'] == 'Sin Límite':
        best_depth = None
    else:
        best_depth = int(config['max_depth'])
        
    print(f"🔬 Simulando Split Simple para: Dataset='{dataset_name}', max_depth={config['max_depth']}")
    
    # --- 4. Cargar los datos de TRAIN de este dataset ---
    train_file = os.path.join(DATA_DIR, f"data_{dataset_name}_TRAIN.csv")
    try:
        df_train = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"  ❌ ¡ERROR! No se encontraron los archivos para '{dataset_name}'")
        continue

    # --- 5. Preparar datos (X, y) ---
    if 'map_name' in df_train.columns:
        df_train = df_train.drop(columns=['map_name'])
        
    X_train_full = df_train.drop(columns=['action'])
    y_train_full = df_train['action']

    # --- 6. Simular el método "antiguo": un simple train_test_split ---
    # Partimos el dataset de TRAIN (8 mapas) en 80% y 20%
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    print(f"  > Datos de TRAIN divididos (80/20).")

    # --- 7. Entrenar el modelo SÓLO en el 80% ---
    model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    model.fit(X_train_split, y_train_split)
    
    # --- 8. Probar el modelo SÓLO en el 20% ---
    y_pred = model.predict(X_test_split)

    # --- 9. Calcular el Accuracy de este split simple ---
    simple_split_accuracy = accuracy_score(y_test_split, y_pred)

    print(f"    > Accuracy (CV de 10 folds):     {config['cv_accuracy']:.4f}")
    print(f"    > Accuracy (Split Simple 80/20): {simple_split_accuracy:.4f}\n")

    # Guardar estos resultados
    comparison_results.append({
        "dataset": dataset_name,
        "max_depth": config['max_depth'],
        "cv_accuracy": config['cv_accuracy'],
        "simple_split_accuracy": simple_split_accuracy
    })

print("=======================================================")
print("✅ ¡Comparación completada!")
print("=======================================================")

# --- 10. Guardar la tabla de comparación ---
if comparison_results:
    df_comparison = pd.DataFrame(comparison_results)
    
    results_file = os.path.join(RESULTS_DIR, "comparison_cv_vs_split.csv")
    df_comparison.to_csv(results_file, index=False)
    
    print(f"🏆 Tabla de comparación guardada en: {results_file}")
    print(df_comparison.head())
else:
    print("⚠️ No se generaron resultados de comparación.")

print("\n--- Fin del Ejercicio 2 (Parte 3) ---")
