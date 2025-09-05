import gymnasium as gym
import numpy as np

def qlearning_evaluate_agent(q_table: np.ndarray, episodes: int = 100) -> None:
    """
    Evalúa un agente Q-Learning entrenado en el entorno Taxi-v3.

    Parámetros:
        q_table (np.ndarray): tabla Q entrenada
        episodes (int): número de episodios de evaluación
    
    Retorna:
        None
    
    Muestra estadísticas de desempeño del agente:
        - Promedio de pasos por episodio
        - Promedio de recompensas por episodio
        - Porcentaje de episodios exitosos
    """

    env = gym.make("Taxi-v3", render_mode="human")
    steps = []
    rewards = []
    successes = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        while not done:
            action = np.argmax(q_table[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            if reward == 20: successes.append(1)
        
        steps.append(step)
        rewards.append(total_reward)
    
    print(f"Promedio de pasos por episodio: {np.mean(steps):.2f}")
    print(f"Promedio de recompensas por episodio: {np.mean(rewards):.2f}")
    print(f"Porcentaje de episodios exitosos: {(np.sum(successes) / episodes) * 100:.2f}%")

    env.close()

def qlearning_train_agent(episodes: int, alpha: float, gamma: float, epsilon: float, decay_rate: float) -> np.ndarray:
    """
    Entrena un agente Q-Learning en el entorno Taxi-v3.
    
    Parámetros:
        episodes (int): número de episodios de entrenamiento
        alpha (float): tasa de aprendizaje
        gamma (float): factor de descuento
        epsilon (float): probabilidad inicial de exploración
        decay_rate (float): tasa de decaimiento de epsilon por episodio
    
    Retorna:
        np.ndarray: tabla Q entrenada
    """

    env = gym.make("Taxi-v3")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state, :])
            new_state, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
            total_reward += reward  
            state = new_state
        
        if (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1}: Recompensa Acumulada: {total_reward}, Epsilon: {epsilon:.2f}")

        epsilon = max(0.01, epsilon * decay_rate)

    env.close()
    return q_table

def main():
    print("Iniciando entrenamiento del agente Q-Learning...")
    q_table = qlearning_train_agent(episodes=2000, alpha=1.0, gamma=0.99, epsilon=1.0, decay_rate=0.995)
    print("Entrenamiento completado.")
    print("\nEvaluando el agente entrenado...")
    qlearning_evaluate_agent(q_table, episodes=100)
    print("Evaluación completada.")


if __name__ == "__main__":
    main()