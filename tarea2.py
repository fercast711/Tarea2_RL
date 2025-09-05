import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def random_train_agent(episodes: int, seeds: np.ndarray) -> None:
    """
    Entrena un agente aleatorio en el entorno Taxi-v3.
    
    Parámetros:
        episodes (int): número de episodios de entrenamiento
        seeds (np.ndarray): array de semillas para inicializar el entorno en cada episodio
    Retorna:
        None
    """

    env = gym.make("Taxi-v3")
    for episode in range(episodes):
        seed = seeds[episode] if seeds is not None else None
        _ , _ = env.reset(seed=int(seed))
        done = False
        total_reward = 0.0 # Recompensa acumulada del episodio

        while not done:
            action = env.action_space.sample() # Política aleatoria
            state, reward, terminated, truncated, _  = env.step(action) 
            done = terminated or truncated
            total_reward += reward
        
        if (episode + 1) % 100 == 0: # Mostrar progreso cada 100 episodios
            print(f"Episodio {episode + 1}: Recompensa Acumulada: {total_reward}") 

    env.close()

def qlearning_train_agent(episodes: int, alpha: float, gamma: float, epsilon: float, decay_rate: float, seeds: np.ndarray) -> np.ndarray:
    """
    Entrena un agente Q-Learning en el entorno Taxi-v3.
    
    Parámetros:
        episodes (int): número de episodios de entrenamiento
        alpha (float): tasa de aprendizaje
        gamma (float): factor de descuento
        epsilon (float): probabilidad inicial de exploración
        decay_rate (float): tasa de decaimiento de epsilon por episodio
        seeds (np.ndarray): array de semillas para inicializar el entorno en cada episodio
    Retorna:
        np.ndarray: tabla Q entrenada
    """

    env = gym.make("Taxi-v3")
    n_states = env.observation_space.n # 500 estados
    n_actions = env.action_space.n # 6 acciones

    q_table = np.zeros((n_states, n_actions)) # Inicializar tabla Q en ceros

    for episode in range(episodes):
        seed = seeds[episode] if seeds is not None else None
        state, _ = env.reset(seed=int(seed))
        done = False
        total_reward = 0.0 # Recompensa acumulada del episodio

        while not done:
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state, :]) # Politica epsilon-greedy
            new_state, reward, terminated, truncated, _  = env.step(action) 
            done = terminated or truncated
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]) # Actualización de la tabla Q
            total_reward += reward  
            state = new_state
        
        if (episode + 1) % 100 == 0: # Mostrar progreso cada 100 episodios
            print(f"Episodio {episode + 1}: Recompensa Acumulada: {total_reward}, Epsilon: {epsilon:.2f}") 

        epsilon = max(0.01, epsilon * decay_rate) # Decaimiento de epsilon, mínimo 0.01

    env.close()
    return q_table


def evaluate_agent(agent_type: str, q_table: np.ndarray = None, episodes: int = 100, seeds: np.ndarray = None, render: str = None) -> None:
    """
    Evalúa un agente Q-Learning entrenado en el entorno Taxi-v3.

    Parámetros:
        agent_type (str): tipo de agente a evaluar ("random" o "qlearning")
        q_table (np.ndarray, opcional): tabla Q entrenada (requerida si agent_type es "qlearning")
        episodes (int): número de episodios de evaluación
        seeds (np.ndarray): array de semillas para inicializar el entorno en cada episodio
        render (str, opcional): modo de renderizado del entorno (por ejemplo, "human" para visualización)
    Retorna:
        None
    
    Muestra estadísticas de desempeño del agente:
        - Promedio de pasos por episodio
        - Promedio de recompensas por episodio
        - Porcentaje de episodios exitosos
    """

    env = gym.make("Taxi-v3", render_mode=render)
    steps = [] # Cantidad de pasos por episodio
    rewards = [] # Recompensa total por episodio
    successes = [] # Episodios exitosos
    rewards_mean = [] # Recompensa media por episodio
    for ep in range(episodes):
        seed = seeds[ep] if seeds is not None else None
        state, _ = env.reset(seed=int(seed))
        done = False
        step = 0.0 # Contador de pasos dados en el episodio
        total_reward = 0.0 # Recompensa acumulada del episodio
        while not done:
            if agent_type == "random":
                action = env.action_space.sample() # Politica aleatoria
            elif agent_type == "qlearning" and q_table is not None:
                action = np.argmax(q_table[state, :]) # Politica greedy
            else:
                raise ValueError("Tipo de agente no válido o tabla Q no proporcionada para Q-Learning.")
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1.0
            
        if reward == 20: successes.append(1) # Episodio exitoso
        steps.append(step) # Guardar cantidad de pasos del episodio
        rewards.append(total_reward) # Guardar recompensa total del episodio
        rewards_mean.append(total_reward / step) # Guardar recompensa media del episodio
    
    reward_mean = np.mean(rewards) # Promedio de recompensas por episodio
    success_rate = (np.sum(successes) / episodes) # Porcentaje de episodios exitosos
    steps_mean = np.mean(steps) # Promedio de pasos por episodio

    print(f"Promedio de pasos por episodio: {steps_mean:.2f}")
    print(f"Promedio de recompensas por episodio: {reward_mean:.2f}")
    print(f"Porcentaje de episodios exitosos: {success_rate * 100:.2f}%")

    env.close()

    return {
        'success_rate': success_rate,
        'avg_reward': rewards_mean,
        'avg_steps': steps_mean
    }

def plot_line(rewards_mean1, rewards_mean2, label1, label2, title, xlabel, ylabel):
    sns.set_style(style="whitegrid")
    sns.lineplot(data=rewards_mean1, label=label1)
    sns.lineplot(data=rewards_mean2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_bar(metrics1, metrics2, label1, label2, title, xlabel, ylabel):
    sns.set_style(style="whitegrid")
    sns.barplot(x=[label1, label2], y=[metrics1, metrics2])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    np.random.seed(44)
    seeds = np.random.randint(0, 10000, size=2000)

    print("Iniciando entrenamiento del agente Q-Learning...\n")
    q_table = qlearning_train_agent(episodes=2000, alpha=1.0, gamma=0.99, epsilon=1.0, decay_rate=0.995, seeds=seeds)
    print("\nEntrenamiento completado.")

    print("\nIniciando entrenamiento del agente Aleatorio...\n")
    random_train_agent(episodes=2000, seeds=seeds)
    print("\nEntrenamiento completado.")

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=100)
    print("\nEvaluando los agentes entrenados...")
    print("\nEvaluando agente Q-Learning:")
    metrics_q = evaluate_agent("qlearning", q_table, episodes=100, seeds=seeds)
    print("\nEvaluando agente Aleatorio:")
    metrics_random = evaluate_agent("random", episodes=100, seeds=seeds)
    print("\nEvaluación completada.")

    plot_line(metrics_q["avg_reward"], metrics_random["avg_reward"], "Q-Learning", "Random", "Recompensa Media por Episodio", "Episodios", "Recompensa Media")
    plot_bar(metrics_q["avg_steps"], metrics_random["avg_steps"], "Q-Learning", "Random", "Longitud promedio por Episodio", "Tipo de Agente", "Longitud Promedio")
    plot_bar(metrics_q["success_rate"]*100, metrics_random["success_rate"]*100, "Q-Learning", "Random", "Porcentaje de Episodios Exitosos", "Tipo de Agente", "Porcentaje de Éxito (%)")


if __name__ == "__main__":
    main()