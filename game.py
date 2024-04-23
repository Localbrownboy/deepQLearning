from learner_agent import Agent
from game_env import SnakeEnv
import gym

# Setup
INPUT_DIMS = 8  # some big number that represents the state
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=128, n_actions=4, lr=0.0001, input_dims=[INPUT_DIMS])
NUM_GAMES = 1000
# env = SnakeEnv()
env = gym.make("LunarLander-v2")

if __name__ == '__main__':
    for i in range(NUM_GAMES):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation=observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

            # Render the game state to the console or GUI
            env.render(mode='human')  # Ensure this is a non-blocking call if using GUI

            # Optionally clear the screen between renders if using console
            # This depends on your OS; example for Unix-like systems:
            # os.system('clear')  # You might use 'cls' on Windows or handle this in `render`

        print(f'Game {i + 1}/{NUM_GAMES}, Score: {score}')
