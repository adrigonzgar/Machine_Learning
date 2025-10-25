#######################
#     registry.py     #
#######################
# Registra un entorno Taxi personalizado con el mapa definido en map_1.txt

import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.envs.registration import register
from map_loader import prepare_for_env
import numpy as np

# Cargar el mapa desde el archivo


class CustomTaxiEnv(TaxiEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        desc_list = [[char.encode() for char in row] for row in MAP]
        self.desc = np.array(desc_list)

# Registrar el entorno con un ID nuevo
register(
    id="Taxi-Custom-v1",
    entry_point="registry:CustomTaxiEnv",
    max_episode_steps=200
)


