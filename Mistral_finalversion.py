#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import matplotlib.pyplot as plt
from vllm import LLM,SamplingParams
import numpy as np
import re
import random
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import gc

# Model paths for different LLMs to be tested
model_path_qwen ="/gpfs/projects/bsc100/models/science/Qwen/Qwen2.5-7B-Instruct"
model_path_llama ="/gpfs/projects/bsc100/models/meta-llama/Llama-3.1-8B-Instruct"
model_path_mistral="/gpfs/projects/bsc100/models/Mistral/Mistral-7B-Instruct-v0.3"

# Sampling parameters for different types of responses
params_explain = SamplingParams(temperature = 0.8, max_tokens = 1000)  # Higher temp for creative explanations
params_extract = SamplingParams(temperature = 0.3, max_tokens = 50)    # Lower temp for precise extraction

# Step-by-step reasoning instructions for the LLM
steps = """
1. There are four possible outcomes in this game, depending on your choice (A or B) and the other player's choice (A or B).
2. Group the outcomes based on your decision:
   - If you choose A: (A,A) and (A,B)
   - If you choose B: (B,A) and (B,B)
3. Analyze and compare the outcomes in each group step by step.
4. Make sure your analysis is consistent with the rules of the game and the possible outcomes.
5. Based on your analysis, choose either A or B.
"""

def extract_choice(text):
    """
    Extract A or B choice from LLM response text
    Returns: (choice, is_valid) tuple
    """
    # Remove punctuation and clean text
    text = re.sub(r'[^\w\s]', '', text.strip())
    
    words = text.split()

    # Check for unambiguous choice
    if 'A' in words and 'B' not in words:
        return 'A', True
    elif 'B' in words and 'A' not in words:
        return 'B', True
    else:
        return 'Unknown', False

def extract_good_bad(text):
    """
    Extract quality assessment (good/bad) from validation response
    """
    text = re.sub(r'[^\w\s]', '', text.lower().strip())

    words = text.split()

    if 'good' in words and 'bad' not in words:
        return 'good'
    elif 'bad' in words and 'good' not in words:
        return 'bad'
    else:
        return 'unknown'

def convert_messages_to_prompt_llama(messages):
    """
    Convert message format to Llama-specific prompt format
    """
    prompt = "<|begin_of_text|>\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def convert_messages_to_prompt_qwen(messages):
    """
    Convert message format to Qwen-specific prompt format
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def convert_messages_to_prompt_mistral(messages):
    """
    Convert message format to Mistral-specific prompt format
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == 'user':
            prompt += '<s>[INST] ' + content.strip() + ' [/INST]'
        elif role == 'assistant':
            prompt += ' ' + content.strip() + ' </s>'
    return prompt

def save_matrix(matrix,name):
    """
    Save matrix as heatmap visualization
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, extent=[5, 15, 0, 10], aspect='auto')
    plt.xlabel("T")
    plt.ylabel("S")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(name+".png")

def load_llm(model_path):
    """
    Load LLM model with specific configuration
    """
    return LLM(
        model=model_path,
        dtype="float16",
        trust_remote_code=True,
        download_dir=model_path
    )

# Game instructions - emphasizes one-shot nature, simultaneity, and monetary incentives
instructions_script_short = """
        This one-shot game, is made of only one round with an anonymous player. You will play only once.

        To play you must choose one of two options: A and B, the other player will also choose between A and B. Both players
        are subjected to the same rules.

        You won't know the other player decision until the end of the round so you won't be able to change your choice
        after knowing the other player's decision.
        The other player won't know your decision until the end of the round so he won't be able to change his choice
        after knowing yours.

        You will be playing simultaneously with the other player.

        Both players will receive a monetary prize. The monetary prize for the player will be
        the amount of points the player has earned multiplied by 10.

        For example, if player 1 earns 9 points and player 2 earns 5 points. 
        Player 1 will receive 9*10 = 90 euros and player 2 will receive 5*10= 50 euros.

        The monetary prize of each player only depends on his number of points.
        The difference in points of both players has no effect on the prize.

        Both players will receive the prize, regardless of who earned more or less money.
        The outcome is solely determined by the number of points earned by each player.

        """

# Initialize game parameter ranges
S = np.arange(0,11)     # Sucker's payoff values (0-10)
T = np.arange(5,16)     # Temptation payoff values (5-15)
S_harmony = np.arange(6,11)  # Harmony game S values
T_harmony = np.arange(5,10)  # Harmony game T values
game_matrix = np.zeros([11,11])  # Results matrix (11x11 for all S,T combinations)

# Generate all possible game configurations
all_values_script_0 = []  # Version 0: A=cooperate, B=defect
game_order = []           # Track (s,t) parameter pairs

for s in S:
    for t in T:
        # Standard format: A is cooperative choice (10,10), B is selfish
        values_script = f"""
        If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
        If you choose A and the other player chooses B. You earn {s} points, the other player earns {t} points.
        If you choose B and the other player chooses A. You earn {t} points, the other player earns {s} points.
        If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.
        """
        all_values_script_0.append(values_script)
        game_order.append([s,t])

# Generate alternative version with A/B labels swapped
all_values_script_1 = []  # Version 1: B=cooperate, A=defect
for s in S:
    for t in T:
        # Swapped format: B is now cooperative choice
        values_script = f"""
        If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
        If you choose B and the other player chooses A. You earn {s} points, the other player earns {t} points.
        If you choose A and the other player chooses B. You earn {t} points, the other player earns {s} points.
        If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.
        """
        all_values_script_1.append(values_script)

# Main experimental loop
repetitions = 10  # Number of times to repeat each game

for rep in range(repetitions):
    
    condition = False  # Loop termination condition
    condition_ignore_log = False  # Skip quality checking if stuck
    games_to_play = [x for x in range(121)]  # All 121 games (11x11 parameter combinations)
    iter_loop = 0  # Track iterations for debugging
    
    # Continue until all games are completed
    while condition is False:
        
        print('rep:', rep, 'games:', len(games_to_play), 'iter_logic:', iter_loop)
        
        long_answers = []  # Store detailed reasoning responses
        random_list = []   # Track which version (0 or 1) was used
        
        # Load model for generating explanations
        llm_long = load_llm(model_path_mistral)
        
        num_games = len(games_to_play)
        
        # Generate responses for each remaining game
        for game in games_to_play:
            
            random_number = np.random.rand()  # Randomly choose version 0 or 1
            
            s = game_order[game][0]  # Current s parameter
            t = game_order[game][1]  # Current t parameter
            
            # Randomly use version 0 (A=cooperate) or version 1 (B=cooperate)
            if random_number <= 0.5:
                random_list.append(1)
                prompt1 = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": instructions_script_short},
                    {"role": "user", "content": all_values_script_0[game] + steps},]
                
                prompt1_mod = convert_messages_to_prompt_mistral(prompt1)
                outputs = llm_long.generate(prompt1_mod, params_explain)
                answer1 = outputs[0].outputs[0].text
                long_answers.append(answer1)
                
            elif random_number > 0.5:
                random_list.append(2)
                prompt1 = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": instructions_script_short},
                    {"role": "user", "content": all_values_script_1[game] + steps},]
                
                prompt1_mod = convert_messages_to_prompt_mistral(prompt1)
                outputs = llm_long.generate(prompt1_mod, params_explain)
                answer1 = outputs[0].outputs[0].text
                long_answers.append(answer1)
        
        # Clean up memory
        del llm_long
        gc.collect()
        
        # Load different model for quality checking and choice extraction
        llm_short = load_llm(model_path_qwen)
        
        games_to_play_copy = games_to_play.copy()
        
        # Process each generated response
        for ans_index in range(len(games_to_play)):
            
            game = games_to_play[ans_index]
            s = game_order[game][0]
            t = game_order[game][1]
            answer1 = long_answers[ans_index]
            quality = 'good'  # Default assumption
            
            # Quality checking phase (unless skipped)
            if condition_ignore_log == False:
                
                # Determine which game version was used
                if random_list[ans_index] == 1:
                    points = all_values_script_0[game]
                else:
                    points = all_values_script_1[game]
                
                # Comprehensive quality checking prompt with examples
                message = f"""
                You have to read the answer of Player 1 and output good or bad according to the following criteria:

                Criteria for good:
                - ALL arithmetic/mathematic comparisons and claims are CORRECT.
                - ALL descriptions of the possible outcomes are CORRECT.
                - ONLY TWO possible outputs per choice, for A: (A,A) and (A,B), for B: (B,A) and (B,B).
                - The final choice is CONSISTENT with the previous reasoning.
                - ALL statements make sense.
                - GOOD UNDERSTANDING of the rules.

                Criteria for bad:
                - ONE OR MORE arithmetic/mathematic comparisons or claims are WRONG.
                - ONE OR MORE descriptions of the possible outcomes are WRONG.
                - Does not understand that there are ONLY TWO possible outcomes per choice.
                - Final choice is NOT CONSISTENT with the previous reasoning.
                - ONE OR MORE statements do not make sense.
                - BAD UNDERSTANDING of the rules.

                Rules of the game:
                {instructions_script_short}

                Example 1, bad reasoning:

                Rules of Example 1:
                
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 7 points, the other player earns 6 points.
                If you choose B and the other player chooses A. You earn 6 points, the other player earns 7 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.

                Answer of Example 1:

                Based on the rules and possible outcomes, here's the analysis:

                - If you choose A, you can earn either 10 points (if the other player chooses A) or 7 points (if the other player chooses B).
                - If you choose B, you can earn either 5 points (if the other player chooses B) or 6 points (if the other player chooses A).

                However, the rules state that the difference in points between you and the other player has no effect on the prize. Therefore, it comes down to the following:

                - If you choose A, the highest possible prize is 10 points, but the lowest possible prize is 7 points.
                - If you choose B, the highest possible prize is 5 points, but the lowest possible prize is 6 points.

                Since the goal is to maximize the possible prize, it would seem that choosing A gives you the potential for a higher prize (100 euros) compared to choosing B (60 euros as the highest potential prize). However, it's important to keep in mind that the other player is making the same analysis as well, so it might be wiser to choose B to reduce the chances of both players choosing A and thus reducing the potential prize for both.

                As this is a one-shot game, it's difficult to say for certain which choice will lead to a higher average prize for both players. It could be a game of bluff and strategy rather than a purely mathematical decision.

                In this analysis, I would lean towards choosing B, to potentially lower the chances of both players receiving a lower prize, but it's ultimately up to your personal decision.

                Output of Example 1: bad

                Example 2, good reasoning:

                Rules of Example 2:
                
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 10 points, the other player earns 6 points.
                If you choose B and the other player chooses A. You earn 6 points, the other player earns 10 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.

                Answer of Example 2:
            
                To make a decision, let's analyze the outcomes based on the rules and possible outcomes:

                1. If you choose A:
                   - (A,A): Both players earn 10 points each.
                   - (A,B): You earn 10 points, and the other player earns 6 points.

                2. If you choose B:
                   - (B,A): You earn 6 points, and the other player earns 10 points.
                   - (B,B): Both players earn 5 points each.

                3. Since the monetary prize for each player is the number of points multiplied by 10, it is clear that earning more points results in a higher prize.

                4. Comparing the outcomes in each group:
                   - If you choose A: (A,A) gives you the highest reward, and (A,B) gives you the same reward but a lower-one for the other player.
                   - If you choose B: (B,B) gives you the lowest reward, and (B,A) gives a slightly higher reward, but still lower than all possible rewards when choosing A.

                5. Based on this information, choosing A would result in a higher potential reward, making it the better choice.
                Even though one might worry about the other player choice, it doesn't make sense since the only possible outcome of choosing A is 10 points, which is the maximum. 
                The other player choice isn't going to affect your outcome at all, there is no risk.

                Output of Example 2: good

                Example 3, bad reasoning:

                Rules of Example 3:

                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 9 points, the other player earns 15 points.
                If you choose B and the other player chooses A. You earn 15 points, the other player earns 9 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.

                Answer of Example 3:

                Let's analyze the outcomes: 
                - If you choose A, the probability of the other player also choosing A is 50%. 
                In this case, you would earn 10 points, and the other player would also earn 10 points (outcome A,A). 
                The other possibility is that the other player chooses B, in which case you would earn 9 points, and they would earn 15 points (outcome A,B). 
                - If you choose B, the probability of the other player also choosing B is 25%. 
                In this case, you would earn 5 points, and the other player would also earn 5 points (outcome B,B). 
                The other possibility is that the other player chooses A, in which case you would earn 15 points, and they would earn 9 points (outcome B,A). 

                Now, let's compare the outcomes: 
                - If you choose A, you have two possible outcomes: one with a higher reward (10 points) and one with a lower reward (9 points). 
                - If you choose B, you have two possible outcomes: one with a lower reward (5 points) and one with a higher reward (15 points). 

                Since you don't know the other player's choice, it's impossible to determine the exact outcome. 
                However, it's interesting to note that if the other player chooses A, choosing A would result in a lower reward (9 points) compared to choosing B (15 points). 
                On the other hand, if the other player chooses B, choosing A would result in a higher reward (10 points) compared to choosing B (5 points). 
                Given this analysis, it seems that choosing B gives you the potential for a higher reward, but it's also possible that the other player will choose A in which case you would have a lower reward. 
                It's truly a gamble. In this game, there is no optimal strategy as the game relies on the other player's choice, which remains unknown. 
                Therefore, it's hard to make a definitive choice between A and B as there is no clear advantage in either option. 
                Based on the analysis and the lack of a perfect answer, I'll leave the choice up to you. 
                You can choose either A or B as there's no guaranteed outcome. Good luck!

                Output of Example 3: bad

                Example 4, good reasoning:

                Rules of Example 4:

                If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
                If you choose B and the other player chooses A. You earn 8 points, the other player earns 5 points.
                If you choose A and the other player chooses B. You earn 5 points, the other player earns 8 points.
                If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.

                Answer of Example 4:

                Based on the given information, I will analyze the outcomes for each choice:

                If you choose A:
                - If the other player chooses A, you both will receive 5 points each.
                - If the other player chooses B, you will receive 5 points, while the other player receives 8 points.

                If you choose B:
                - If the other player chooses A, you will receive 8 points, while the other player receives 5 points.
                - If the other player chooses B, you both will receive 10 points each.

                If you choose B, the maximum amount of points you can get is 10 points and the minimum, 8 points.
                If you choose A the maximum amount of points you can get is 5 points and the minimum, 5 points. 
                The maximum points is greater when you choose B, leading to a higher prize. 
                The minimum points is greater when you choose B, leading to a lower risk of getting little points. 
                Then, choice B is optimal since it is the option with the highest reward and the lower risk.

                Output of Example 4: good

                Example 5, bad reasoning:

                Rules of Example 5:

                If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
                If you choose B and the other player chooses A. You earn 10 points, the other player earns 7 points.
                If you choose A and the other player chooses 7. You earn 6 points, the other player earns 10 points.
                If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.

                Answer of Example 5:

                Given the rules, choosing A would bring a maximum of 5 points, and choosing B would bring a maximum of 10 points if the other player also chooses B. 
                Since we don't know the other player's choice, it's impossible to guarantee a higher score by choosing A or B. 
                However, if we're forced to make a choice, choosing B seems like the better option since it has a higher potential reward if the other player also chooses B. 
                But, ultimately, this game is a gamble, and both choices have an element of risk.

                Outcome of Example 5: bad

                Example 6, good reasoning:

                Rules of Example 6:

                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 1 points, the other player earns 15 points.
                If you choose B and the other player chooses A. You earn 15 points, the other player earns 1 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.

                Answer of Example 6:

                Choosing A can lead to the worst result (earning only 1 point) and choosing B can lead to the best result (earning 15 points).
                Then I choose B to earn at least 5 points and also have the potential to the highest reward.

                Output of Example 6: good

                Rules of game to analyze:
                {points}

                Answer of Player 1, to analyze:
                {answer1}

                Important:
                - Output ONLY one word: good or bad
                - Do not add punctuation, extra spaces, or explanations.
                """

                # Generate quality assessment
                prompt3 = [
                {"role": "system", "content": "You are a logical verifier. Your purpose is to look for inconsistencies and errors in a given text."},
                {"role": "user", "content": message}]

                prompt3_mod = convert_messages_to_prompt_qwen(prompt3)
                outputs = llm_short.generate(prompt3_mod, params_extract)
                answer3 = outputs[0].outputs[0].text
                quality = extract_good_bad(answer3)
            
            # Process response if quality is good OR quality checking is disabled
            if (quality == 'good') or (condition_ignore_log == True):
                
                # Extract the actual choice (A or B) from the reasoning
                prompt2 = [
                    {"role": "system", "content": "You're a helpful assistant."},
                    {"role": "user", "content": "The player who was asked to choose between A and B answered " + answer1},
                    {"role": "user", "content": 'What did the person who wrote the message chose? Answer shortly.'}]
                prompt2_mod = convert_messages_to_prompt_qwen(prompt2)
                outputs = llm_short.generate(prompt2_mod, params_extract)
                answer2 = outputs[0].outputs[0].text
                choice, state = extract_choice(answer2)
                
                # Record result in matrix if choice extraction was successful
                if random_list[ans_index] == 1:  # Version 0 was used
                    if state == True:
                        game_delete = games_to_play[ans_index]
                        games_to_play_copy.remove(game_delete)
                        
                        # Convert choice to cooperation rate (A=cooperate=1, B=defect=0)
                        if choice == 'A':
                            game_matrix[10-s,t-5] += 1  # Cooperation
                        elif choice == 'B':
                            game_matrix[10-s,t-5] += 0  # Defection
                            
                elif random_list[ans_index] == 2:  # Version 1 was used (A/B swapped)
                    if state == True:
                        game_delete = games_to_play[ans_index]
                        games_to_play_copy.remove(game_delete)
                        
                        # Reverse mapping since A/B were swapped
                        if choice == 'A':
                            game_matrix[10-s,t-5] += 0  # Defection (A was defect in version 1)
                        elif choice == 'B':
                            game_matrix[10-s,t-5] += 1  # Cooperation (B was cooperate in version 1)
        
        # Update games remaining and check termination conditions
        games_to_play = games_to_play_copy
        num_games_now = len(games_to_play)
        
        # If no progress made, skip quality checking to avoid infinite loop
        if num_games_now == num_games:
            condition_ignore_log = True
        
        # Check if all games completed
        if games_to_play == []:
            condition = True
        else:
            iter_loop += 1
            
        # Clean up memory
        del llm_short
        gc.collect()

# Convert raw counts to cooperation rates by dividing by number of repetitions
game_matrix = game_matrix/repetitions

# Save final results to file
np.savetxt('mistral10rep_final.txt', game_matrix, fmt = '%.2f')
