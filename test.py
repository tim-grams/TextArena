""" A minimal script showing how to run textarena locally """

import textarena as ta 

agents = {
    0: ta.agents.HumanAgent(),
    1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-001"),
}

# initialize the environment
env = ta.make(env_id="SimpleTak-v0-train")
env.reset(num_players=len(agents))

def single_game(env, agents):
    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)
    rewards, game_info = env.close()
    return rewards, game_info

n_samples = 250
for i in range(n_samples):
    rewards, game_info = single_game(env, agents)
    print(f"Game {i+1} rewards: {rewards}")
    print(f"Game {i+1} game info: {game_info}")

