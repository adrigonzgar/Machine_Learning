import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import warnings
import numpy as np

# Suprimir warnings
warnings.filterwarnings("ignore")

print("--- Iniciando Cross-Validation (EJ 2.1) ---")

# --- Configuración ---
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
output_path = os.path.join(RESULTS_DIR, "cv_results_table.csv")

# Lista de nuestros 4 datasets
datasets_files = {
    "original": "data_original_TRAIN.csv",
    "new_features": "data_new_features_TRAIN.csv",
    "kbest": "data_kbest_TRAIN.csv",
    "rfe": "data_rfe_TRAIN.csv"
}

# Parámetros a probar (profundidades del árbol)
max_depths = [10, 15, 20, None] # None = "sin límite"
all_results = []

# --- 1. Iterar por cada uno de los 4 datasets ---
for name, filename in datasets_files.items():
    print(f"\n--- Procesando Dataset: '{name}' ---")
    
    # 1. Cargar datos
    try:
        # --- ¡ARREGLO CLAVE! Usamos el separador por defecto (COMA) ---
        file_path = os.path.join(DATA_DIR, filename)
        df_train = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⚠️ ¡Aviso! No se encontró el archivo {filename}. Saltando...")
        continue
    except pd.errors.EmptyDataError:
        print(f"⚠️ ¡Aviso! El archivo {filename} está vacío. Saltando...")
        continue

    # 2. Preparar datos
    if 'map_name' in df_train.columns:
        df_train = df_train.drop(columns=['map_name'])
    
    if df_train.empty:
        print(f"⚠️ ¡Aviso! Datos vacíos para '{name}'.")
        continue
        
    X_train = df_train.drop(columns=['action'])
    y_train = df_train['action']
    
    # --- 2. Iterar por cada parámetro (max_depth) ---
    for depth in max_depths:
        depth_str = "None" if depth is None else str(depth) # Convertir a string
        print(f"  Probando max_depth = {depth_str}...")
        
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        
        # 3. Ejecutar Cross-Validation (CV=10)
        cv_scores = cross_validate(
            model, 
            X_train, 
            y_train, 
            cv=10, 
            scoring=['accuracy', 'precision_weighted', 'recall_weighted']
        )
        
        # 4. Guardar los resultados medios
        all_results.append({
            "dataset": name,
            "max_depth": depth_str,
            "mean_accuracy": np.mean(cv_scores['test_accuracy']),
            "mean_precision": np.mean(cv_scores['test_precision_weighted']),
            "mean_recall": np.mean(cv_scores['test_recall_weighted'])
        })

# --- 3. Guardar la tabla final de resultados ---
if not all_results:
    print("\n❌ ¡ERROR! No se generaron resultados. Comprueba los archivos de datos.")
else:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="mean_accuracy", ascending=False)
    
    # --- ¡ARREGLO CLAVE! Guardar con el separador por defecto (COMA) ---
    results_df.to_csv(output_path, index=False)
    
    print("\n--- ¡Cross-Validation Completada! ---")
    print(f"✅ Tabla de resultados guardada en: {output_path}")
    print("\n🏆 Top 5 Mejores Modelos (según CV):")
    print(results_df.head(5))

