import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def executar(episodios, em_treinamento=True, renderizar=False):
    # Inicializa o ambiente 'Taxi-v3'
    ambiente = gym.make('Taxi-v3', render_mode='human' if renderizar else None)

    # Inicializa a matriz Q
    if em_treinamento:
        q = np.zeros((ambiente.observation_space.n, ambiente.action_space.n))  # Matriz 500 x 6
    else:
        with open('taxi.pkl', 'rb') as arquivo:
            q = pickle.load(arquivo)

    taxa_aprendizado = 0.9  # Taxa de aprendizado (alpha)
    fator_desconto = 0.9  # Fator de desconto (gamma)
    epsilon = 1  # Probabilidade inicial de ações aleatórias (exploração)
    taxa_decrescimento_epsilon = 0.0001  # Taxa de decaimento do epsilon
    rng = np.random.default_rng()  # Gerador de números aleatórios

    recompensas_por_episodio = np.zeros(episodios)

    for i in range(episodios):
        estado = ambiente.reset()[0]  # Reseta o ambiente e obtém o estado inicial
        terminado = False  # Indica se o episódio terminou
        truncado = False  # Indica se o episódio foi truncado (limite de passos)
        recompensa_total = 0

        while not terminado and not truncado:
            # Escolha da ação: exploração ou exploração
            if em_treinamento and rng.random() < epsilon:
                acao = ambiente.action_space.sample()  # Ação aleatória
            else:
                acao = np.argmax(q[estado, :])  # Ação baseada na matriz Q

            # Executa a ação no ambiente
            novo_estado, recompensa, terminado, truncado, _ = ambiente.step(acao)
            recompensa_total += recompensa

            # Atualiza a matriz Q durante o treinamento
            if em_treinamento:
                q[estado, acao] = q[estado, acao] + taxa_aprendizado * (
                    recompensa + fator_desconto * np.max(q[novo_estado, :]) - q[estado, acao]
                )

            estado = novo_estado

        # Decaimento do epsilon para reduzir ações aleatórias ao longo do tempo
        epsilon = max(epsilon - taxa_decrescimento_epsilon, 0)

        # Ajusta a taxa de aprendizado quando epsilon chega a zero
        if epsilon == 0:
            taxa_aprendizado = 0.0001

        recompensas_por_episodio[i] = recompensa_total

    ambiente.close()

    # Plota a soma das recompensas por episódio
    soma_recompensas = np.zeros(episodios)
    for t in range(episodios):
        soma_recompensas[t] = np.sum(recompensas_por_episodio[max(0, t-100):(t+1)])
    plt.plot(soma_recompensas)
    plt.xlabel('Episódios')
    plt.ylabel('Soma das Recompensas (média dos últimos 100 episódios)')
    plt.savefig('taxi.png')

    # Salva a matriz Q em um arquivo para uso futuro
    if em_treinamento:
        with open("taxi.pkl", "wb") as arquivo:
            pickle.dump(q, arquivo)

if __name__ == '__main__':
    # Treina o agente com 15.000 episódios
    executar(15000)

    # Avalia o agente treinado em 10 episódios com renderização
    executar(10, em_treinamento=False, renderizar=True)
