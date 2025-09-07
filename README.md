# Tarea 2: Aprendizaje por Refuerzo

Este proyecto implementa y compara dos agentes para el entorno Taxi-v3 de OpenAI Gymnasium:

- **Agente Aleatorio**: Selecciona acciones al azar.
- **Agente Q-Learning**: Aprende una política óptima mediante el algoritmo Q-Learning.

## Archivos

- `tarea2.py`: Código principal con funciones de entrenamiento, evaluación y visualización.
- `requirements.txt`: Lista de dependencias necesarias.
- `Tarea2_RL-1.pdf`: Enunciado de la tarea.

## Ejecución

1. Crear entorno virtual
	```bash
	python -m venv venv
	```

2. Activar entorno virtual en windows
	```bash
	.\venv\Scripts\activate
	```

3. Instala las dependencias:
	```bash
	pip install -r requirements.txt
	```

4. Ejecuta el script principal:
	```bash
	python tarea2.py
	```

## Descripción del experimento

Se entrena un agente Q-Learning y un agente aleatorio durante 2000 episodios cada uno. Luego, ambos agentes se evalúan en 100 episodios usando las mismas semillas para garantizar la comparabilidad.

Se reportan y grafican:
- Recompensa media por episodio
- Longitud promedio de los episodios
- Porcentaje de episodios exitosos

## Resultados esperados

El agente Q-Learning debería superar ampliamente al agente aleatorio en todas las métricas, mostrando mayor recompensa media, menor longitud de episodio y mayor porcentaje de éxito.

## Dependencias

- numpy
- gymnasium
- pygame
- matplotlib
- seaborn

## Notas

- El código es reproducible gracias al uso de semillas aleatorias.
- Los gráficos se muestran al final de la ejecución.

## Autor

Fernando Castillo y Gerardo Diaz