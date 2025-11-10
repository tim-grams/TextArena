""" A minimal script showing how to run textarena locally """

import textarena as ta 

agents = {
    0: ta.agents.OpenRouterAgent(model_name="qwen/qwen3-8b"),
    1: ta.agents.OpenRouterAgent(model_name="qwen/qwen3-8b")
}

# initialize the environment
env = ta.make(env_id="IteratedPrisonersDilemma-v0-train")
# env = ta.wrappers.SimpleRenderWrapper(env=env) #, render_mode="standard")
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
  player_id, observation = env.get_observation()
  print('#### OBSERVATION ####')
  print(observation)
  action = agents[player_id](observation)
  print('#### ACTION ####')
  print(player_id, action)
  done, step_info = env.step(action=action)
rewards, game_info = env.close()
print(rewards)
print(game_info)
