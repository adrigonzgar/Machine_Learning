#######################
#     registry.py     #
#######################
import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.envs.registration import register
from map_loader import prepare_for_env
import numpy as np

# Esta variable global 'MAP' se rellenará desde los scripts EX_1
MAP = [] 

class CustomTaxiEnv(TaxiEnv):
    def __init__(self, render_mode=None):
        
        # --- ¡ESTE ES EL ORDEN CORRECTO! ---

        # 1. Preparamos el mapa en formato NumPy ANTES de nada
        desc_list = [[char.encode() for char in row] for row in MAP]
        desc_array = np.array(desc_list)

        # 2. Ahora sí, llamamos a la clase madre y le pasamos 
        #    nuestro mapa (desc_array) usando el parámetro 'desc'
        super().__init__(render_mode=render_mode, desc=desc_array)


# Registrar el entorno con un ID nuevo
register(
    id="Taxi-Custom-v1",
    entry_point="registry:CustomTaxiEnv",
    max_episode_steps=200
)
