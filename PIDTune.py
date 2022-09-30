# Importar bibliotecas e funcoes
import SAC
import argparse
import configparser
import random
import os
import numpy as np 
import tensorflow as tf

def main():
    # Construir uma interface de comando de linha para alteracao dos parametros das redes
    parser = argparse.ArgumentParser("Configuracões do Sistema")
    parser.add_argument('--config', type=str, default='config.config')
    args = parser.parse_args()

    # Inicializacao dos parametros das redes
    # Configuracao inicial do replay buffer
    config - configparser.RawConfigParser()
    config.read(args.config)
    gamma = config.getfloat('train', 'gamma')
    buffer_size = config.getint('train', 'buffer_size')
    learning_rate = config.getfloat('train','learning_rate')
    batch_size = config.getint('train','batch_size')
    episodes = config.getint('train','eps')
    step = config.getint('train', 'step')
    goal = config.getint('train', 'score_goal')
    path = config.getstring('Save Model', 'path')

    # Env -> Enviroment -> Dados do robô simulado ou real-time
    # Env fornece os estados (y), acoes (kp, ki, kd), sinal de erro (e(t)) ...
    policy = SAC(env, gamma, config, buffer_size, learning_rate)

    for eps in range(episode):
        score = 0
        max_value = 0
        mean_value = 0
        error = 0
        state = env[0]

        for i in range(step):
            action = policy.select_action(state)  # Selecao da acao com base no estado
            Nstate, reward, aval = env[]
            policy.buffer.push(state, action, reward, Nstate, aval)
            state = Nstate

            score += reward
            if aval or score == goal:
                policy.save_model(path)
                break
        
    
if __name__ == "__main__":
    main()