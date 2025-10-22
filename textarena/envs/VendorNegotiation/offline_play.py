""" A script for running VendorNegotiation with full game logging """

import textarena as ta 
import os
import json
import datetime

agents = {
    0: ta.agents.AWSBedrockAgent(model_id='anthropic.claude-3-5-sonnet-20241022-v2:0',region_name='us-west-2'),
    1: ta.agents.AWSBedrockAgent(model_id='anthropic.claude-3-5-sonnet-20241022-v2:0',region_name='us-west-2'),
}

agents = {
    0: ta.agents.HumanAgent(),
    1: ta.agents.HumanAgent(),
}

# Initialize game log
game_log = {
    "game_metadata": {
        "env_id": "VendorNegotiation-v0",
        "timestamp": datetime.datetime.now().isoformat(),
        "agent_types": {str(k): type(v).__name__ for k, v in agents.items()}
    },
    "turns": [],
    "final_results": {}
}

# initialize the environment
env = ta.make(env_id="VendorNegotiation-v0")
env.reset(num_players=len(agents))

# Capture initial game metadata
game_log["game_metadata"].update({
    "num_products": env.num_products,
    "max_rounds": env.max_rounds,
    "selected_products": env.selected_products,
    "brand_target": env.brand_target,
    "vendor_baseline": env.vendor_baseline,
    "brand_role": env.brand_role_name,
    "vendor_role": env.vendor_role_name,
    "allowed_discounts": env.allowed_discounts,
    "num_simulations": env.num_simulations
})

# main game loop
done = False 
turn_counter = 0
while not done:
    turn_counter += 1
    
    # Get current game state
    board_state = env.get_board_str()
    
    # Get observation and action
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    
    # Log this turn
    turn_log = {
        "turn": turn_counter,
        "player_id": player_id,
        "observation": observation,
        "action": action,
        "board_state_before": board_state
    }
    
    # Execute the action
    done, step_info = env.step(action=action)
    
    # Add post-action state
    turn_log["board_state_after"] = env.get_board_str()
    turn_log["done"] = done
    turn_log["step_info"] = step_info
    
    game_log["turns"].append(turn_log)

# Show final results after game ends
final_board_state = ""
if done:
    print("\n" + "="*60)
    print("FINAL GAME RESULTS")
    print("="*60)
    final_board_state = env.get_board_str()
    print(final_board_state)
    print("="*60)

rewards, game_info = env.close()

# Capture final results
game_log["final_results"] = {
    "final_board_state": final_board_state,
    "rewards": rewards,
    "game_info": game_info,
    "total_turns": turn_counter
}

print(f"\nRewards: {rewards}")
print(f"Game Info: {game_info}")

# Save game log to JSON file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log/vendor_negotiation_log_{timestamp}.json"
with open(log_filename, 'w') as f:
    json.dump(game_log, f, indent=2)

print(f"\nFull game log saved to: {log_filename}")
