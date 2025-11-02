import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.exceptions import NotFittedError

# --- Configuración ---
INPUT_DIR = "data/processed"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Los 3 mejores modelos que seleccionamos en el Paso 2
TOP_3_MODELS = [
    {"name": "rfe", "max_depth": 10},
    {"name": "new_features", "max_depth": 10},
    {"name": "original", "max_depth": 10} 
]

print("--- Entrenando y Evaluando Modelos Finales ---")

report_data = []

for config in TOP_3_MODELS:
    dataset_name = config["name"]
    depth = config["max_depth"]
    model_id = f"{dataset_name}_depth_{depth}"
    
    print(f"\nProcessing Model: {model_id}")

    # --- 1. Cargar Datos ---
    try:
        # Los archivos generados por feature_selection.py se llaman así
        if dataset_name == "rfe":
            train_file = os.path.join(INPUT_DIR, "data_rfe_TRAIN.csv")
            test_file = os.path.join(INPUT_DIR, "data_rfe_TEST.csv")
        elif dataset_name == "kbest": # Lo dejamos por si acaso
             train_file = os.path.join(INPUT_DIR, "data_kbest_TRAIN.csv")
             test_file = os.path.join(INPUT_DIR, "data_kbest_TEST.csv")
        else: # Para 'original' y 'new_features'
            train_file = os.path.join(INPUT_DIR, f"data_{dataset_name}_TRAIN.csv")
            test_file = os.path.join(INPUT_DIR, f"data_{dataset_name}_TEST.csv")

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
    except FileNotFoundError as e:
        print(f"  ❌ ¡ERROR! No se encontró el archivo: {e.filename}")
        print("  Asegúrate de haber ejecutado los scripts 'combine_data.py' y 'feature_selection.py' primero.")
        continue

    # --- 2. Preparar X (features) e y (target) ---
    TARGET_COLUMN = 'action'
    
    # Quitamos 'map_name' si existe, y el target 'action'
    features_to_drop = [TARGET_COLUMN, 'map_name']
    
    # Obtenemos la lista de features dinámicamente
    feature_columns = [col for col in df_train.columns if col not in features_to_drop]
    
    X_train = df_train[feature_columns]
    y_train = df_train[TARGET_COLUMN]
    
    X_test = df_test[feature_columns]
    y_test = df_test[TARGET_COLUMN]

    print(f"  Features usadas: {feature_columns}")

    # --- 3. Entrenar Modelo ---
    print(f"  Entrenando DecisionTreeClassifier (max_depth={depth})...")
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # --- 4. Evaluar en Test ---
    print("  Evaluando en el Test set...")
    try:
        y_pred = model.predict(X_test)
        
        # Generar reporte de clasificación
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n--- Resultados en TEST ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("--------------------------")
        
        report_data.append({
            "model_id": model_id,
            "test_accuracy": accuracy,
            "test_precision_macro": report_dict["macro avg"]["precision"],
            "test_recall_macro": report_dict["macro avg"]["recall"]
        })

    except NotFittedError:
        print("  ❌ ¡ERROR! El modelo no fue entrenado.")
        continue
    except Exception as e:
        print(f"  ❌ ¡ERROR durante la evaluación! {e}")
        continue

    # --- 5. Guardar Modelo Entrenado ---
    model_filename = os.path.join(OUTPUT_DIR, f"model_{model_id}.joblib")
    joblib.dump(model, model_filename)
    print(f"  ✅ Modelo guardado en: {model_filename}")


print("\n\n--- Resumen de Evaluación Final (Test Set) ---")
if report_data:
    df_report = pd.DataFrame(report_data)
    print(df_report.to_markdown(index=False))
else:
    print("No se generaron reportes. Revisa los errores.")