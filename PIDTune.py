# Importar bibliotecas e funcoes
import SAC
import argparse
import configparser
import random
import os
import numpy as np 
import tensorflow as tf


def main():

    parser = argparse.ArgumentParser("Configuracões do Sistema")
    parser.add_argument('--config', type=str, default='config.config')
    args = parser.parse_args()

    # Inicializacao dos parametros das redes
    config - configparser.RawConfigParser()
    config.read(args.config)
    gamma = config.getfloat('train', 'gamma')
    buffer_size = config.getint('train', 'buffer_size')
    learning_rate = config.getfloat('train','learning_rate')
    batch_size=config.getint('train','batch_size')
    episodes=config.getint('train','eps')

    policy = SAC(env, gamma, config, buffer_size, learning_rate)
    

    # Configuracao inicial do replay buffer

    for i in range(episode):
        #
        A =i
        
    
if __name__ == "__main__":
    main()