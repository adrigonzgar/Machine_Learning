from pickle import TRUE
import gymnasium as gym
import pygame
import joblib
import pandas as pd
import numpy as np
from registry import prepare_for_env, CustomTaxiEnv
import registry  # 👈 registra Taxi-Custom-v1
import os
import time

# 💡💡💡 ¡FIX 1: Silenciar los UserWarning para que no se congele! 💡💡💡
# Le decimos a Python que ignore este tipo específico de aviso.
import warnings
# 'UserWarning' es una clase base de Python, no necesitamos importarla.
warnings.filterwarnings("ignore", category=UserWarning)
# 💡💡💡

# =====================================================
# 🧮 1. COPIAR FUNCIONES DE FEATURES DE EX_1.PY
# =====================================================

locs = [(0,0),(0,4),(4,0),(4,3)] # R, G, Y, B

def decode(env, obs):
    """
    Decodifica la observación (un número) en sus partes legibles.
    (taxi_r, taxi_c, pass_idx, dest_idx)
    """
    # FIX: Accedemos al entorno "desenvuelto" para encontrar .decode()
    return tuple(env.unwrapped.decode(obs))


def manhattan_distance(coord1, coord2):
    """Calcula la distancia de Manhattan entre dos (r, c)"""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def get_rel_dir(taxi_coord, target_coord):
    """
    Calcula la dirección relativa de un objetivo.
    0:N, 1:S, 2:E, 3:W, 4:On Target
    """
    if taxi_coord == target_coord:
        return 4 # Está en el objetivo
    
    delta_r = taxi_coord[0] - target_coord[0]
    delta_c = taxi_coord[1] - target_coord[1]
    
    if abs(delta_r) > abs(delta_c):
        return 0 if delta_r > 0 else 1 # 0:Norte, 1:Sur
    else:
        return 3 if delta_c > 0 else 2 # 3:Oeste, 2:Este

def get_all_features(env, obs):
    """
    Una función maestra que calcula TODAS las features
    posibles (originales y nuevas)
    """
    decoded_state = decode(env, obs)
    taxi_r, taxi_c, pass_idx, dest_idx = decoded_state
    
    # Coordenadas del taxi
    taxi_coord = (taxi_r, taxi_c)
    
    # 0:R, 1:G, 2:Y, 3:B, 4:In Taxi
    if pass_idx == 4:
        passenger_in_taxi = 1
        pass_coord = taxi_coord # Pasajero está en el taxi
    else:
        passenger_in_taxi = 0
        pass_coord = locs[pass_idx] # Pasajero está en una de las 4 locs
    
    # Coordenadas de destino
    dest_coord = locs[dest_idx]
    
    # --- Features Originales ---
    feats_original = {
        'taxi_r': taxi_r,
        'taxi_c': taxi_c,
        'pass_r': pass_coord[0],
        'pass_c': pass_coord[1],
        'dest_r': dest_coord[0],
        'dest_c': dest_coord[1],
        'passenger_in_taxi': passenger_in_taxi
    }
    
    # --- Features Nuevas ---
    d_taxi_pass = manhattan_distance(taxi_coord, pass_coord)
    d_taxi_dest = manhattan_distance(taxi_coord, dest_coord)

    # Si el pasajero está en el taxi, la distancia al pasajero es irrelevante
    if passenger_in_taxi == 1:
        d_taxi_pass = -1 # Valor especial
        rel_dir_taxi_to_pass = -1 # Valor especial
    else:
        rel_dir_taxi_to_pass = get_rel_dir(taxi_coord, pass_coord)
        
    rel_dir_taxi_to_dest = get_rel_dir(taxi_coord, dest_coord)

    feats_new = {
        'd_taxi_pass': d_taxi_pass,
        'd_taxi_dest': d_taxi_dest,
        'rel_dir_taxi_to_pass': rel_dir_taxi_to_pass,
        'rel_dir_taxi_to_dest': rel_dir_taxi_to_dest,
        'passenger_in_taxi': passenger_in_taxi
    }

    # Combinar todos los diccionarios
    all_features = {**feats_original, **feats_new}
    return all_features


# =====================================================
# 🖊️ 2. Definir las features que espera CADA modelo
# =====================================================
# ¡El ORDEN debe ser EXACTAMENTE el mismo que en el entrenamiento!

FEATURES_ORIGINAL = [
    'taxi_r', 'taxi_c', 'pass_r', 'pass_c', 'dest_r', 'dest_c', 'passenger_in_taxi'
]

FEATURES_NEW = [
    'd_taxi_pass', 'd_taxi_dest', 'rel_dir_taxi_to_pass', 
    'rel_dir_taxi_to_dest', 'passenger_in_taxi'
]

# Basado en tu output de Ej. 2, RFE se quedó con estas 3
FEATURES_RFE = [
    'd_taxi_pass', 'd_taxi_dest', 'rel_dir_taxi_to_pass'
]

# =====================================================
# 🚀 3. Función principal de despliegue
# =====================================================

def run_simulation(model_path, map_file, headless=False):
    """
    Ejecuta una simulación del agente en un mapa.
    
    Args:
        model_path: Ruta al modelo .joblib
        map_file: Nombre del archivo del mapa en la carpeta maps/
        headless: Si True, no muestra la ventana de Pygame (útil para modo automático)
    
    Returns:
        int: Número de pasos si el agente resuelve el mapa correctamente
        None: Si el agente falla o se trunca el episodio
    """
    
    # --- Cargar Modelo ---
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el modelo {model_path}")
        return None
    
    # --- Cargar Mapa ---
    try:
        map_path = os.path.join("maps", map_file)
        registry.MAP = prepare_for_env(map_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el mapa {map_file} en la carpeta 'maps/'")
        return None

    # --- Determinar qué features usar ---
    if "original" in model_path:
        feature_list = FEATURES_ORIGINAL
        if not headless:
            print("🧠 Usando features: ORIGINALES")
    elif "rfe" in model_path:
        feature_list = FEATURES_RFE
        if not headless:
            print("🧠 Usando features: RFE")
    elif "new_features" in model_path:
        feature_list = FEATURES_NEW
        if not headless:
            print("🧠 Usando features: NEW_FEATURES")
    else:
        print(f"❌ Error: No se reconoce el tipo de modelo '{model_path}'")
        return None

    # --- Crear Entorno ---
    render_mode = "human" if not headless else "rgb_array"  # rgb_array no abre ventanas
    env = gym.make("Taxi-Custom-v1", render_mode=render_mode)
    obs, info = env.reset()
    clock = pygame.time.Clock() if not headless else None
    
    done = False
    truncated = False
    total_steps = 0
    max_steps = 200 # Límite para evitar bucles infinitos

    while not done and not truncated:
        total_steps += 1
        
        try:
            # 1. Obtener todas las features del estado actual
            current_state_features = get_all_features(env, obs)
            
            # 💡💡💡 ¡FIX 2: La solución correcta (DataFrame)! 💡💡💡
            # Pasamos un DataFrame de 1 fila con
            # nombres de columnas, en lugar de una lista.
            # 💡💡💡
            
            # 2. Filtrar solo las features que este modelo necesita
            features_dict = {feat: current_state_features[feat] for feat in feature_list}
            
            # 3. Convertir a DataFrame de 1 fila
            X_predict = pd.DataFrame(features_dict, index=[0])
            
            # 4. Predecir la acción
            action = model.predict(X_predict)[0]
            
            # 5. Ejecutar la acción
            obs, reward, done, truncated, info = env.step(action)
            
            # 6. Limitar los pasos
            if total_steps >= max_steps:
                truncated = True
                if not headless:
                    print(f"Límite de {max_steps} pasos alcanzado.")

            # 7. Renderizar (y ralentizar para que se pueda ver)
            if not headless:
                env.render()
                clock.tick(10) # 10 frames por segundo

        except Exception as e:
            print(f"\n❌ ¡ERROR DURANTE LA SIMULACIÓN! ❌")
            print(e)
            # Imprimir más detalles del error
            import traceback
            traceback.print_exc()
            truncated = True # Forzar la salida del bucle


    # --- Reporte Final ---
    if not headless:
        print("\n--- ¡Simulación Terminada! ---")
        if done and not truncated:
            print(f"✅✅ ÉXITO ✅✅")
            print(f"Mapa '{map_file}' resuelto en {total_steps} pasos.")
        else:
            print(f"❌❌ FRACASO ❌❌")
            if truncated:
                 print(f"El agente no resolvió el mapa. Límite de {max_steps} pasos alcanzado o error.")
            else:
                 print(f"El agente falló por otra razón.")
    
    result = total_steps if (done and not truncated) else None
    
    env.close()
    
    if not headless:
        print("---------------------------------")
    
    return result


# =====================================================
# 🖥️ 4. Menú Interactivo
# =====================================================
if __name__ == "__main__":
    
    # ===========================================
    # 🎛️ MODO AUTOMÁTICO (Ejercicio 3)
    # ===========================================
    AUTO_MODE = True  # Cambiar a True para activar modo automático
    
    if AUTO_MODE:
        print("🚀 MODO AUTOMÁTICO ACTIVADO")
        print("="*60)
        
        # Definir los 3 modelos finales
        model_files = [
            "model_original_depth_10.joblib",
            "model_new_features_depth_10.joblib",
            "model_rfe_depth_10.joblib"
        ]
        
        # Obtener todos los mapas
        maps_dir = "maps"
        available_maps = [f for f in os.listdir(maps_dir) if f.endswith(".txt")]
        available_maps.sort()  # Ordenar para consistencia
        
        print(f"\n📊 Ejecutando {len(model_files)} modelos en {len(available_maps)} mapas...")
        print(f"Total de simulaciones: {len(model_files) * len(available_maps)}")
        print("="*60)
        
        # Lista para almacenar resultados
        results = []
        
        # Ejecutar cada modelo en cada mapa
        for model_file in model_files:
            model_path = os.path.join("models", model_file)
            
            for map_file in available_maps:
                print(f"\n🤖 Modelo: {model_file} | Mapa: {map_file}")
                print("-" * 60)
                
                # Ejecutar simulación en modo headless
                steps = run_simulation(model_path, map_file, headless=True)
                
                # Imprimir progreso simplificado
                if steps is not None:
                    print(f"✅ Resuelto en {steps} pasos")
                else:
                    print(f"❌ Falló")
                
                # Guardar resultado
                results.append({
                    'model': model_file,
                    'map': map_file,
                    'steps': steps
                })
                
                # Pausa entre simulaciones
                time.sleep(1)
        
        # Crear DataFrame con los resultados
        results_df = pd.DataFrame(results)
        
        # Crear directorio results si no existe
        os.makedirs("results", exist_ok=True)
        
        # Guardar a CSV
        csv_path = "results/deploy_results.csv"
        results_df.to_csv(csv_path, index=False)
        print("\n" + "="*60)
        print(f"✅ Resultados guardados en: {csv_path}")
        
        # Imprimir resumen
        print("\n📈 RESUMEN DE RESULTADOS")
        print("="*60)
        
        for model_file in model_files:
            model_results = results_df[results_df['model'] == model_file]
            solved = model_results[model_results['steps'].notna()]
            total_maps = len(model_results)
            solved_count = len(solved)
            avg_steps = solved['steps'].mean() if len(solved) > 0 else None
            
            print(f"\n🤖 {model_file}:")
            print(f"   Mapas resueltos: {solved_count}/{total_maps}")
            if avg_steps is not None:
                print(f"   Promedio de pasos: {avg_steps:.2f}")
            else:
                print(f"   Promedio de pasos: N/A")
        
        print("\n" + "="*60)
        print("✅ Ejecución automática completada")
        print("="*60)
        
    else:
        # --- Seleccionar Modelo ---
        print("🤖 Modelos disponibles para desplegar:")
        models_dir = "models"
        available_models = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
        
        # Filtrar solo los 3 que nos interesan
        final_models = [m for m in available_models if "original" in m or "rfe" in m or "new_features" in m]
        
        if not final_models:
            print(f"❌ Error: No se encontraron modelos en la carpeta '{models_dir}'")
            print("Asegúrate de haber ejecutado 'train_final_models.py' del Ejercicio 2.")
            exit()

        for i, m in enumerate(final_models):
            print(f"  [{i}] {m}")
        
        try:
            model_idx = int(input("👉 Elige el número del modelo a probar: "))
            selected_model = os.path.join(models_dir, final_models[model_idx])
        except Exception as e:
            print("Selección inválida. Saliendo.")
            exit()

        # --- Seleccionar Mapa ---
        print("\n🗺️ Mapas disponibles para probar:")
        maps_dir = "maps"
        available_maps = [f for f in os.listdir(maps_dir) if f.endswith(".txt")]
        for i, m in enumerate(available_maps):
            print(f"  [{i}] {m}")

        try:
            map_idx = int(input("👉 Elige el número del mapa a probar: "))
            selected_map = available_maps[map_idx]
        except Exception as e:
            print("Selección inválida. Saliendo.")
            exit()
        
        # --- Ejecutar ---
        print(f"\n🚀 Desplegando {selected_model} en {selected_map}...")
        print("Cierra la ventana de Pygame para terminar.")
        time.sleep(2)
        
        run_simulation(selected_model, selected_map)