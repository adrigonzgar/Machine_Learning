#####################################
#   EX_2_cross_validation.py        #
#                                   #
#  Entrena y evalúa los 4 datasets  #
#  usando Cross-Validation (CV=10)  #
#####################################

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# --- Configuración ---
# Las 4 "familias" de datasets que creamos en el Ej. 1
DATASET_NAMES = ["original", "new_features", "kbest", "rfe"]
DATA_DIR = "data/processed"
RESULTS_DIR = "results" # Carpeta para guardar la tabla de resultados

# Parámetros del Árbol de Decisión que queremos probar
# Probaremos árboles con distintas profundidades máximas
# None significa "sin límite" (crecerá todo lo que quiera)
DEPTHS_TO_TRY = [5, 10, 15, None]

# Número de "folds" (partes) para la Cross-Validation
CV_FOLDS = 10 
# ---------------------

# Crear la carpeta de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# Lista para guardar todos los resultados de todas las pruebas
all_results = []

print("--- Iniciando Ejercicio 2: Entrenamiento con Cross-Validation ---")

# --- 1. Bucle principal: Probar cada uno de los 4 datasets ---
for dataset_name in DATASET_NAMES:
    print(f"\n=======================================================")
    print(f"🔬 Probando Dataset: '{dataset_name}'")
    print(f"=======================================================")
    
    # Construir la ruta al archivo de entrenamiento
    train_file = os.path.join(DATA_DIR, f"data_{dataset_name}_TRAIN.csv")
    
    try:
        df_train = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"❌ ¡ERROR! No se encontró el archivo: {train_file}")
        print("Asegúrate de haber completado el Ejercicio 1.")
        continue
    except pd.errors.EmptyDataError:
        print(f"❌ ¡ERROR! El archivo está vacío: {train_file}")
        continue
        
    # --- 2. Preparar datos (X, y) ---
    # Quitamos 'map_name' si existe, no es una feature
    if 'map_name' in df_train.columns:
        df_train = df_train.drop(columns=['map_name'])
        
    X_train = df_train.drop(columns=['action'])
    y_train = df_train['action']
    
    print(f"  > Datos cargados: {len(X_train)} muestras, {len(X_train.columns)} features.")

    # --- 3. Bucle interno: Probar cada parámetro (max_depth) ---
    for depth in DEPTHS_TO_TRY:
        
        depth_str = "Sin Límite" if depth is None else str(depth)
        print(f"\n  ---------------------------------")
        print(f"  🌳 Entrenando con max_depth = {depth_str}")
        
        # Crear el modelo de Árbol de Decisión
        # Usamos random_state=42 para que los resultados sean reproducibles
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        
        # --- 4. Ejecutar Cross-Validation (como pide la guía) ---
        # Lo hacemos 3 veces, una para cada métrica
        
        # Métrica 1: Accuracy (la más común)
        # 'accuracy' es la métrica por defecto
        cv_accuracy = cross_val_score(model, X_train, y_train, 
                                      cv=CV_FOLDS, scoring='accuracy')
        
        # Métrica 2: Precision (macro)
        # 'macro' calcula la métrica para cada acción (0-5) y hace la media
        cv_precision = cross_val_score(model, X_train, y_train, 
                                       cv=CV_FOLDS, scoring='precision_macro')
        
        # Métrica 3: Recall (macro)
        cv_recall = cross_val_score(model, X_train, y_train, 
                                    cv=CV_FOLDS, scoring='recall_macro')

        # --- 5. Calcular la media de las 10 rondas (folds) ---
        # cross_val_score devuelve un array (ej. [0.95, 0.92, ...]), calculamos la media
        mean_accuracy = np.mean(cv_accuracy)
        mean_precision = np.mean(cv_precision)
        mean_recall = np.mean(cv_recall)

        print(f"    > Resultados de CV (media de {CV_FOLDS} folds):")
        print(f"    > Accuracy:   {mean_accuracy:.4f}")
        print(f"    > Precision:  {mean_precision:.4f}")
        print(f"    > Recall:     {mean_recall:.4f}")

        # Guardar estos resultados en nuestra lista
        all_results.append({
            "dataset": dataset_name,
            "max_depth": depth_str,
            "cv_accuracy": mean_accuracy,
            "cv_precision": mean_precision,
            "cv_recall": mean_recall
        })

print("\n=======================================================")
print("✅ ¡Entrenamiento completado!")
print("=======================================================")

# --- 6. Guardar la tabla de resultados ---
if all_results:
    # Convertir la lista de resultados en una tabla (DataFrame)
    df_results = pd.DataFrame(all_results)
    
    # Ordenar la tabla para ver los mejores (mayor accuracy) primero
    df_results = df_results.sort_values(by="cv_accuracy", ascending=False)
    
    # Guardar la tabla en un archivo CSV
    results_file = os.path.join(RESULTS_DIR, "cv_results_table.csv")
    df_results.to_csv(results_file, index=False)
    
    print(f"🏆 Tabla de resultados guardada en: {results_file}")
    print("\n--- MEJORES MODELOS (según CV Accuracy) ---")
    print(df_results.head())
else:
    print("⚠️ No se generaron resultados.")

print("\n--- Fin del Ejercicio 2 (Parte 1) ---")
