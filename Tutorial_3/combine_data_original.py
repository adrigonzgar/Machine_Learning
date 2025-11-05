#############################
#  combine_data_original.py #
#############################
import pandas as pd
import os

print("Iniciando combinación de datasets ORIGINALES...")

# --- ¡DECIDID VUESTRA DIVISIÓN AQUÍ! ---
# Usad la MISMA división que antes (8 mapas para tren, 2 para test)
TRAIN_MAPS = [
    "map_1.txt", "map_2.txt", "map_3.txt", "map_4.txt",
    "map_5.txt", "map_6.txt", "map_7.txt", "map_8.txt"
]
TEST_MAPS = [
    "map_9.txt", "map_10.txt"
]
# ----------------------------------------

# --- ¡CAMBIOS AQUÍ! ---
INPUT_DIR = "data/original"           # Leemos de la nueva carpeta
OUTPUT_DIR = "data/processed"
FILE_PREFIX = "dataset_original_"     # Buscamos los nuevos archivos
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def combine_files(map_list, output_filename):
    """
    Lee todos los CSV de una lista de mapas y los junta en un solo archivo.
    """
    print(f"Combinando archivos para: {output_filename}...")
    
    all_dataframes = []
    
    for map_name in map_list:
        # map_1.txt -> dataset_original_map_1.csv
        file_name = f"{FILE_PREFIX}{map_name.replace('.txt', '')}.csv"
        file_path = os.path.join(INPUT_DIR, file_name)
        
        try:
            df_map = pd.read_csv(file_path)
            all_dataframes.append(df_map)
            print(f"  ✅ Cargado: {file_name} ({len(df_map)} filas)")
        except FileNotFoundError:
            print(f"  ⚠️ ¡AVISO! No se encontró el archivo: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"  ⚠️ ¡AVISO! El archivo está vacío: {file_path}")

    if not all_dataframes:
        print(f"❌ ¡ERROR! No se encontraron datos para {output_filename}")
        return

    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n🎉 ¡Éxito! Archivo combinado guardado en:")
    print(f"   {output_path} (Total: {len(df_combined)} filas)")
    print("-" * 30)

# --- Ejecutar el proceso ---
# 1. Combinar datos de ENTRENAMIENTO (originales)
combine_files(TRAIN_MAPS, "data_original_TRAIN.csv") # <-- Nuevo nombre de salida

# 2. Combinar datos de PRUEBA (originales)
combine_files(TEST_MAPS, "data_original_TEST.csv") # <-- Nuevo nombre de salida

print("\nProceso de combinación ORIGINAL completado.")