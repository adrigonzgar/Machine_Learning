import gymnasium as gym
import time
import pygame
import pandas as pd
import numpy as np
from registry import prepare_for_env, CustomTaxiEnv
import registry  # 👈 registra Taxi-Custom-v1
import os

# =====================================================
# 🗺️ 1. Seleccionar mapa
# =====================================================
time.sleep(1)

maps_dir = "maps"
available_maps = [f for f in os.listdir(maps_dir) if f.endswith(".txt")]
print("🗺️ Mapas disponibles:")
for f in available_maps:
    print(f"  - {f}")

map_file = input("👉 Escribe el nombre del mapa (ej: map_1.txt): ")
map_path = os.path.join(maps_dir, map_file)
registry.MAP = prepare_for_env(map_path)
print(f"✅ Mapa {map_path} cargado correctamente.")

# =====================================================
# 🚕 2. Crear entorno con render gráfico
# =====================================================
env = gym.make("Taxi-Custom-v1", render_mode="human")
obs, info = env.reset()
clock = pygame.time.Clock()

# =====================================================
# 🧮 3. Funciones para extraer features
# =====================================================
locs = [(0,0),(0,4),(4,0),(4,3)]  # posiciones R,G,Y,B

def decode_state(state):
    # La codificación correcta es: (((taxi_row * 5 + taxi_col) * 5) + pass_loc) * 4 + dest_idx
    
    dest_idx = state % 4
    state = state // 4
    
    pass_loc = state % 5
    state = state // 5
    
    taxi_col = state % 5
    state = state // 5
    
    taxi_row = state % 5 # state debería ser 0..4 aquí
    
    # Obtener coordenadas de los índices
    dest_row, dest_col = locs[dest_idx]
    if pass_loc < 4:
        pass_row, pass_col = locs[pass_loc]
        in_taxi = 0
    else:
        pass_row, pass_col = -1, -1 # El pasajero está en el taxi
        in_taxi = 1
    
    return {
        "taxi_r": taxi_row,
        "taxi_c": taxi_col,
        "pass_r": pass_row,
        "pass_c": pass_col,
        "dest_r": dest_row,
        "dest_c": dest_col,
        "passenger_in_taxi": in_taxi
    }

def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def rel_dir(src,dst):
    dr,dc = dst[0]-src[0], dst[1]-src[1]
    if abs(dr)>abs(dc): return 1 if dr<0 else -1
    elif abs(dc)>0: return 2 if dc<0 else -2
    return 0

def extract_features(s):
    taxi=(s["taxi_r"],s["taxi_c"])
    dest=(s["dest_r"],s["dest_c"])
    pin=s["passenger_in_taxi"]
    if pin==0:
        passenger=(s["pass_r"],s["pass_c"])
        d_tp=manhattan(taxi,passenger)
        dir_tp=rel_dir(taxi,passenger)
    else:
        d_tp=-1
        dir_tp=0
    d_td=manhattan(taxi,dest)
    dir_td=rel_dir(taxi,dest)
    return {
        "d_taxi_pass": d_tp,
        "d_taxi_dest": d_td,
        "rel_dir_taxi_to_pass": dir_tp,
        "rel_dir_taxi_to_dest": dir_td,
        "passenger_in_taxi": pin
    }

# =====================================================
# 🕹️ 4. Control manual para recoger datos
# =====================================================
print("\n🎮 Controles manuales:")
print(" ↑ = Norte | ↓ = Sur | → = Este | ← = Oeste | P = Pickup | D = Dropoff | Q = Salir")

data = []
running = True

while running:
    # ⚠️ Si la ventana ya se cerró, salimos sin romper nada
    if env.unwrapped.window is None:
        break

    try:
        env.render()
    except pygame.error:
        print("⚠️ Ventana cerrada, saliendo del bucle...")
        break

    action = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 1
            elif event.key == pygame.K_DOWN:
                action = 0
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_LEFT:
                action = 3
            elif event.key == pygame.K_p:
                action = 4
            elif event.key == pygame.K_d:
                action = 5
            elif event.key == pygame.K_q:
                running = False

    if action is not None:
        s_decoded = decode_state(obs)         # <--- ¡ESTO es lo que queremos guardar!
        s_decoded["action"] = action
        s_decoded["map_name"] = map_file   
        data.append(s_decoded)                # <--- Guardamos s_decoded directamente

        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, info = env.reset()

    clock.tick(10)

env.close()

# =====================================================
# 💾 5. Guardar dataset manual automáticamente
# =====================================================
if len(data) > 0:
    df_new = pd.DataFrame(data)
    os.makedirs("data/original", exist_ok=True)  # <-- CAMBIO DE CARPETA
    dataset_name = f"dataset_original_{map_file.replace('.txt','')}.csv" # <-- CAMBIO DE NOMBRE
    dataset_path = os.path.join("data/original", dataset_name) # <-- CAMBIO DE RUTA

    # Si el archivo ya existe, concatenar
    if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
        try:
            df_old = pd.read_csv(dataset_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all.to_csv(dataset_path, index=False)
            print(f"\n📈 {len(df_new)} muestras añadidas a {dataset_name}. Total: {len(df_all)}")
        except pd.errors.EmptyDataError:
            df_new.to_csv(dataset_path, index=False)
            print(f"\n⚠️ Archivo existente vacío. Se sobrescribió con {len(df_new)} muestras.")
    else:
        df_new.to_csv(dataset_path, index=False)
    print(f"\n✅ {len(df_new)} muestras guardadas en nuevo archivo {dataset_name}")


    print("🧾 Columnas:", df_new.columns.tolist())
else:
    print("\n⚠️ No se recolectaron datos.")
