#!/usr/bin/env python
# coding: utf-8

# Game Theory Experiment with Large Language Models
# This script runs a prisoner's dilemma-like game with LLMs to analyze their decision-making patterns

import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
import numpy as np
import re
import random
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import gc

# Model paths for different LLMs
model_path_qwen = "/gpfs/projects/bsc100/models/science/Qwen/Qwen2.5-7B-Instruct"
model_path_llama = "/gpfs/projects/bsc100/models/meta-llama/Llama-3.1-8B-Instruct"
model_path_mistral = "/gpfs/projects/bsc100/models/Mistral/Mistral-7B-Instruct-v0.3"

# Sampling parameters for different types of responses
params_explain = SamplingParams(temperature=0.8, max_tokens=1000)  # For generating explanations
params_extract = SamplingParams(temperature=0.3, max_tokens=50)    # For extracting choices/quality

# Instructions given to the LLM for decision-making process
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
    Extract the choice (A or B) from the LLM's response text
    Returns the choice and a boolean indicating if extraction was successful
    """
    # Remove punctuation and clean the text
    text = re.sub(r'[^\w\s]', '', text.strip())
    words = text.split()

    # Check if only A or only B appears in the text
    if 'A' in words and 'B' not in words:
        return 'A', True
    elif 'B' in words and 'A' not in words:
        return 'B', True
    else:
        return 'Unknown', False

def extract_good_bad(text):
    """
    Extract quality assessment (good/bad) from the LLM's response
    Used to determine if the reasoning was logically sound
    """
    text = re.sub(r'[^\w\s]', '', text.lower().strip())
    words = text.split()

    # Check if only 'good' or only 'bad' appears in the text
    if 'good' in words and 'bad' not in words:
        return 'good'
    elif 'bad' in words and 'good' not in words:
        return 'bad'
    else:
        return 'unknown'

# Prompt formatting functions for different model architectures
def convert_messages_to_prompt_llama(messages):
    """Convert message format to Llama-specific prompt format"""
    prompt = "<|begin_of_text|>\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def convert_messages_to_prompt_qwen(messages):
    """Convert message format to Qwen-specific prompt format"""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def convert_messages_to_prompt_mistral(messages):
    """Convert message format to Mistral-specific prompt format"""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == 'user':
            prompt += '<s>[INST] ' + content.strip() + ' [/INST]'
        elif role == 'assistant':
            prompt += ' ' + content.strip() + ' </s>'
    return prompt

def save_matrix(matrix, name):
    """Save the game results matrix as a heatmap visualization"""
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, extent=[5, 15, 0, 10], aspect='auto')
    plt.xlabel("T")
    plt.ylabel("S")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(name + ".png")

def load_llm(model_path):
    """Load and initialize a language model with specified parameters"""
    return LLM(
        model=model_path,
        dtype="float16",
        trust_remote_code=True,
        download_dir=model_path
    )

# Game instructions presented to the LLM
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
        Player 1 will receive 9*10 = 90 euros and player 2 will receive 10*5= 50 euros.

        The monetary prize of each player only depends on his number of points.
        The difference in points of both players has no effect on the prize.

        Both players will receive the prize, regardless of who earned more or less money.
        The outcome is solely determined by the number of points earned by each player.

        """

# Initialize game parameters
S = np.arange(0, 11)        # S values (sucker's payoff range)
T = np.arange(5, 16)        # T values (temptation payoff range)
S_harmony = np.arange(6, 11)  # Harmony game S values
T_harmony = np.arange(5, 10)  # Harmony game T values
game_matrix = np.zeros([11, 11])  # Matrix to store results (11x11 for all S,T combinations)

# Generate all possible game scenarios
all_values_script_0 = []  # Game scenarios where A=cooperate, B=defect
game_order = []           # Track the (S,T) values for each game

# Create all combinations of S and T values
for s in S:
    for t in T:
        # Game format where A is the cooperative choice
        values_script = f"""
        If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
        If you choose A and the other player chooses B. You earn {s} points, the other player earns {t} points.
        If you choose B and the other player chooses A. You earn {t} points, the other player earns {s} points.
        If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.
        """
        all_values_script_0.append(values_script)
        game_order.append([s, t])

# Create alternative game scenarios where B=cooperate, A=defect
all_values_script_1 = []
for s in S:
    for t in T:
        # Game format where B is the cooperative choice
        values_script = f"""
        If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
        If you choose B and the other player chooses A. You earn {s} points, the other player earns {t} points.
        If you choose A and the other player chooses B. You earn {t} points, the other player earns {s} points.
        If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.
        """
        all_values_script_1.append(values_script)

# Main experimental loop
repetitions = 10  # Number of times to repeat the entire experiment

for rep in range(repetitions):
    print(f"Starting repetition {rep + 1}/{repetitions}")
    
    condition = False  # Loop control variable
    condition_ignore_log = False  # Flag to skip quality checking when stuck
    games_to_play = [x for x in range(121)]  # List of all 121 games (11*11 combinations)
    iter_loop = 0  # Track iteration count within this repetition

    # Continue until all games are completed
    while condition is False:
        print('rep:', rep, 'games:', len(games_to_play), 'iter_logic:', iter_loop)

        long_answers = []  # Store LLM explanations
        random_list = []   # Track which game format was used (1 or 2)

        # Load the LLM for generating explanations (using Llama)
        llm_long = load_llm(model_path_llama)
        num_games = len(games_to_play)

        # Generate responses for each remaining game
        for game in games_to_play:
            random_number = np.random.rand()
            s = game_order[game][0]  # S value for this game
            t = game_order[game][1]  # T value for this game

            # Randomly choose between two game formats to avoid bias
            if random_number <= 0.5:
                # Format 1: A=cooperate, B=defect
                random_list.append(1)
                prompt1 = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": instructions_script_short},
                    {"role": "user", "content": all_values_script_0[game] + steps},
                ]
                prompt1_mod = convert_messages_to_prompt_llama(prompt1)
                outputs = llm_long.generate(prompt1_mod, params_explain)
                answer1 = outputs[0].outputs[0].text
                long_answers.append(answer1)

            else:
                # Format 2: B=cooperate, A=defect
                random_list.append(2)
                prompt1 = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": instructions_script_short},
                    {"role": "user", "content": all_values_script_1[game] + steps},
                ]
                prompt1_mod = convert_messages_to_prompt_llama(prompt1)
                outputs = llm_long.generate(prompt1_mod, params_explain)
                answer1 = outputs[0].outputs[0].text
                long_answers.append(answer1)

        # Clean up memory after generating all explanations
        del llm_long
        gc.collect()
        
        # Load LLM for quality assessment and choice extraction (using Qwen)
        llm_short = load_llm(model_path_qwen)
        games_to_play_copy = games_to_play.copy()

        # Process each generated answer
        for ans_index in range(len(games_to_play)):
            game = games_to_play[ans_index]
            s = game_order[game][0]  # S value for this game
            t = game_order[game][1]  # T value for this game
            answer1 = long_answers[ans_index]
            
            quality = 'good'  # Default quality assessment

            # Quality check (skip if we're ignoring quality due to repeated failures)
            if condition_ignore_log == False:
                # Determine which game format was used
                if random_list[ans_index] == 1:
                    points = all_values_script_0[game]
                else:
                    points = all_values_script_1[game]

                # Comprehensive quality assessment prompt with examples
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

                [Multiple examples of good and bad reasoning follow...]

                Rules of game to analyze:
                {points}

                Answer of Player 1, to analyze:
                {answer1}

                Important:
                - Output ONLY one word: good or bad
                - Do not add punctuation, extra spaces, or explanations.
                """

                # Get quality assessment
                prompt3 = [
                    {"role": "system", "content": "You are a logical verifier. Your purpose is to look for inconsistencies and errors in a given text."},
                    {"role": "user", "content": message}
                ]
                prompt3_mod = convert_messages_to_prompt_qwen(prompt3)
                outputs = llm_short.generate(prompt3_mod, params_extract)
                answer3 = outputs[0].outputs[0].text
                quality = extract_good_bad(answer3)

            # Process answers with good quality or when ignoring quality
            if (quality == 'good') or (condition_ignore_log == True):
                # Extract the actual choice (A or B) from the explanation
                prompt2 = [
                    {"role": "system", "content": "You're a helpful assistant."},
                    {"role": "user", "content": "The player who was asked to choose between A and B answered " + answer1},
                    {"role": "user", "content": 'What did the person who wrote the message chose? Answer shortly.'}
                ]
                prompt2_mod = convert_messages_to_prompt_qwen(prompt2)
                outputs = llm_short.generate(prompt2_mod, params_extract)
                answer2 = outputs[0].outputs[0].text
                choice, state = extract_choice(answer2)

                # Record the result in the matrix based on game format and choice
                if random_list[ans_index] == 1:  # Format 1: A=cooperate
                    if state == True:  # Successfully extracted choice
                        game_delete = games_to_play[ans_index]
                        games_to_play_copy.remove(game_delete)

                        if choice == 'A':  # Chose to cooperate
                            game_matrix[10-s, t-5] += 1  # Add 1 (cooperation)
                        elif choice == 'B':  # Chose to defect
                            game_matrix[10-s, t-5] += 0  # Add 0 (defection)

                elif random_list[ans_index] == 2:  # Format 2: B=cooperate
                    if state == True:  # Successfully extracted choice
                        game_delete = games_to_play[ans_index]
                        games_to_play_copy.remove(game_delete)

                        if choice == 'A':  # Chose to defect
                            game_matrix[10-s, t-5] += 0  # Add 0 (defection)
                        elif choice == 'B':  # Chose to cooperate
                            game_matrix[10-s, t-5] += 1  # Add 1 (cooperation)

        # Update the games list and check progress
        games_to_play = games_to_play_copy
        num_games_now = len(games_to_play)

        # If no progress was made, start ignoring quality checks
        if num_games_now == num_games:
            condition_ignore_log = True

        # Check if all games are completed
        if games_to_play == []:
            condition = True
        else:
            iter_loop += 1
            
        # Clean up memory
        del llm_short
        gc.collect()

# Calculate average cooperation rate across all repetitions
game_matrix = game_matrix / repetitions

# Save the final results matrix
np.savetxt('llama10rep_final.txt', game_matrix, fmt='%.2f')

print("Experiment completed. Results saved to 'llama10rep_final.txt'")
print(f"Final cooperation matrix shape: {game_matrix.shape}")
print(f"Overall cooperation rate: {np.mean(game_matrix):.3f}")
