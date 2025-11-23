#!/usr/bin/env python
# coding: utf-8

"""
Llama_Mistral_finalversion.py
Andrea Cera Palatsi, 2025.

"""

# IMPORT FUNCTIONS
import matplotlib.pyplot as plt
from vllm import LLM,SamplingParams
import numpy as np
import re
import random
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import gc

# DEFINE PATHS OF THE LLMs
model_path_qwen ="/gpfs/projects/bsc100/models/science/Qwen/Qwen2.5-7B-Instruct"
model_path_llama ="/gpfs/projects/bsc100/models/meta-llama/Llama-3.1-8B-Instruct"
model_path_mistral="/gpfs/projects/bsc100/models/Mistral/Mistral-7B-Instruct-v0.3"

# DEFINE PARAMETERS FOR THE LLMs
params_explain = SamplingParams(temperature = 0.8, max_tokens = 1000)
params_extract = SamplingParams(temperature = 0.3, max_tokens = 50)

# LOGICAL STEPS 
steps = """
1. There are four possible outcomes in this game, depending on your choice (A or B) and the other player's choice (A or B).
2. Group the outcomes based on your decision:
   - If you choose A: (A,A) and (A,B)
   - If you choose B: (B,A) and (B,B)
3. Analyze and compare the outcomes in each group step by step.
4. Make sure your analysis is consistent with the rules of the game and the possible outcomes.
5. Based on your analysis, choose either A or B.
"""

# FUNCTIONS
def extract_choice(text):
    """
    Extract choice from short answer.
    
    Args:
        text: short answer of the model
    
    Returns:
        A, B if the extraction was succesful and Unknown if it wasn't
    """
    
    text = re.sub(r'[^\w\s]', '', text.strip())
    
    words = text.split()

    if 'A' in words and 'B' not in words:
        return 'A', True
    elif 'B' in words and 'A' not in words:
        return 'B', True
    else:
        return 'Unknown', False

def extract_good_bad(text):
    """
    Extract veredict of logical verifier.
    
    Args:
        text: output of logical verifier
    
    Returns:
        good if the logical verifier accepts the long answer,
        bad if the logical verifier rejects the long answer,
        unknown if the logical verifier can't make a veredict
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
    Turns dictionary into Llama prompts with the correct structure.
    
    Args:
        messages: dictionary containing prompts in general structure
    
    Returns:
        prompt: prompts in Llama structure
    """
    prompt = "<|begin_of_text|>\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def convert_messages_to_prompt_mistral(messages):
    """
    Turns dictionary into Mistral prompts with the correct structure.
    
    Args:
        messages: dictionary containing prompts in general structure
    
    Returns:
        prompt: prompts in Mistral structure
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

def convert_messages_to_prompt_qwen(messages):
    """
    Turns dictionary into Qwen prompts with the correct structure.
    
    Args:
        messages: dictionary containing prompts in general structure
    
    Returns:
        prompt: prompts in Qwen structure
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def save_matrix(matrix,name):
    """
    Save cooperation matrix as a heatmap visualization
    X-axis: T values (temptation), Y-axis: S values (sucker's payoff)
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, extent = [0, 20, 0, 15], aspect='auto')
    plt.xlabel("T")
    plt.ylabel("S")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(name+".png")

def load_llm(model_path):
    """
    Initialize the LLM with specific configuration for inference
    Using float16 precision and trusting remote code execution
    """
    return LLM(
        model=model_path,
        dtype="float16",
        trust_remote_code=True,
        download_dir=model_path
    )

# INSTRUCTION PROMPT
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

# DEFINE VALUES OF S AND T
S = np.arange(0,21) # Sucker's payoff
T = np.arange(0,21) # Temptation payoff
game_matrix = np.zeros([21,21]) # Matrix that will contain cooperation for the different games
sticking_matrix = np.zeros([21,21]) # Matrix that will contain how many times a game blocks the logical verifier


# CREATES ARRAY THAT CONTAINS ALL THE OUTCOMES OF ALL GAMES
# 0: Corresponds to the case where A is C and B is D
all_values_script_0 = []
game_order = [] # list that contains all combinations of (s,t)
for s in S:
    for t in T:
        values_script = f"""
        If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
        If you choose A and the other player chooses B. You earn {s} points, the other player earns {t} points.
        If you choose B and the other player chooses A. You earn {t} points, the other player earns {s} points.
        If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.
        """
        all_values_script_0.append(values_script)
        game_order.append([s,t])

# 1: Corresponds to the case where A is D and B is C
all_values_script_1 = []
for s in S:
    for t in T:
        values_script = f"""
        If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
        If you choose B and the other player chooses A. You earn {s} points, the other player earns {t} points.
        If you choose A and the other player chooses B. You earn {t} points, the other player earns {s} points.
        If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.
        """
        all_values_script_1.append(values_script)

# DEFINE NUMBER OF ITERATIONS
repetitions =  10

#START OF SIMULATION
for rep in range(repetitions): # rep shows the iteration

    condition = False # when true all games have an asign choice

    condition_ignore_log = False # when true open logical verifier

    games_to_play =  [x for x in range(441)]

    iter_loop = 0 # control how many times the games have been played to have valid answers in one iteration

    while condition is False:

        print('rep:', rep, 'games:', len(games_to_play), 'iter_logic:', iter_loop)

        long_answers = [] # list of long answers
        random_list = [] # meaning of A and B for each long answer

        llm_long = load_llm(model_path_llama) # load llm that will choose

        num_games = len(games_to_play) # games without valid answer, yet

        for game in games_to_play:

            random_number = np.random.rand() # randomly select version 0 or 1 to control for A/B labeling effects

            s = game_order[game][0] # current sucker's payoff
            t = game_order[game][1] # current temptation payoff

            if random_number <= 0.5: # 0 case, A means C and B means D
                random_list.append(1)

                # GENERATE THE PROMPT
                prompt1 = [
                    {"role": "system", "content": "You are a helpful assistant"}, # typical system prompt
                    {"role": "user", "content": instructions_script_short}, # instructions
                    {"role": "user", "content": all_values_script_0[game] + steps},] # game outcomes and logical steps

                prompt1_mod = convert_messages_to_prompt_llama(prompt1) # turn general prompt to specific prompt
                outputs = llm_long.generate(prompt1_mod, params_explain) # generate answer
                answer1 = outputs[0].outputs[0].text # extract answer
                long_answers.append(answer1)

            
            elif random_number > 0.5: # 1 case, A means D and B means C
                random_list.append(2)

                # GENERATE THE PROMPT
                prompt1 = [
                    {"role": "system", "content": "You are a helpful assistant"}, # typical system prompt
                    {"role": "user", "content": instructions_script_short}, # instructions
                    {"role": "user", "content": all_values_script_1[game] + steps},] # game outcomes and logical steps

                prompt1_mod = convert_messages_to_prompt_llama(prompt1) # turn general prompt to specific prompt
                outputs = llm_long.generate(prompt1_mod, params_explain) # generate answer
                answer1 = outputs[0].outputs[0].text # extract answer
                long_answers.append(answer1)

        del llm_long # delete llm from memory
        gc.collect()
        llm_short = load_llm(model_path_qwen) # load llm for extraction

        games_to_play_copy = games_to_play.copy()

        for ans_index in range(len(games_to_play)): # go over all games played

            game = games_to_play[ans_index]

            s = game_order[game][0] #s value of the iteration
            t = game_order[game][1] #t value of the iteration

            answer1 = long_answers[ans_index] # select long answer

            quality = 'good' # if logical verifier gets deactivated, quality per defect is good

            # LOGICAL VERIFIER
            if condition_ignore_log == False: # condition of activation

                if random_list[ans_index] == 1: # check values of A and B
                    points = all_values_script_0[game]
                else:
                    points = all_values_script_1[game]

                # Prompt given to logical verifier llm
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
                Even though one might worry about the other player choice, it doesn’t make sense since the only possible outcome of choosing A is 10 points, which is the maximum. 
                The other player choice isn’t going to affect your outcome at all, there is no risk.

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
                If you choose A and the other player chooses B. You earn 6 points, the other player earns 10 points.
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

                # Prompt in general structure
                prompt3 = [
                {"role": "system", "content": "You are a logical verifier. Your purpose is to look for inconsistencies and errors in a given text."},
                {"role": "user", "content": message}]

                prompt3_mod = convert_messages_to_prompt_qwen(prompt3) # turn prompt to specific structure
                outputs = llm_short.generate(prompt3_mod, params_extract) # generate answer
                answer3 = outputs[0].outputs[0].text # extract answer
                quality = extract_good_bad(answer3) # parse quality assessment 

            # If the long answer passes the logical verifier or if it was deactivated
            # GENERATION OF SHORT ANSWER
            if (quality == 'good') or (condition_ignore_log == True):

                # Generate general prompt to extract short answer from long answer
                prompt2 = [
                    {"role": "system", "content": "You are a helpful assistant."}, # typical system prompt
                    {"role": "user", "content": "The player who was asked to choose between A and B answered " + answer1}, 
                    {"role": "user", "content": 'What did the person who wrote the message chose? Answer shortly.'}]

                prompt2_mod = convert_messages_to_prompt_qwen(prompt2) # adapt prompt for Qwen
                outputs = llm_short.generate(prompt2_mod, params_extract) # generate short answer
                answer2 = outputs[0].outputs[0].text # extract short answer
                choice, state = extract_choice(answer2) # extract choice from short answer

                if random_list[ans_index] == 1: # case 0, A means C and B means D

                    if state == True: # if short answer was extracted

                        game_delete = games_to_play[ans_index] # delete game from list, it has been played correctly
                        games_to_play_copy.remove(game_delete)

                        if choice == 'A': 
                            game_matrix[20-s,t] += 1 # if llm chose to cooperate, add 1 to matrix element

                elif random_list[ans_index] == 2: # case 1, A means D, and B means C

                    if state == True: # if short answer was extracted

                        game_delete = games_to_play[ans_index] # delete game from list, it has been played correctly
                        games_to_play_copy.remove(game_delete)

                        if choice == 'B':
                            game_matrix[20-s,t] += 1 # if llm chose to cooperate, add 1 to matrix element

        games_to_play = games_to_play_copy # update the list with the games left to play correctly
        num_games_now = len(games_to_play) # update number of games left to play

        if num_games_now == num_games: # if in one round any game has been played correctly, deactivate logical verifier
            if condition_ignore_log == False: # we make sure it's the fist time that happens
                condition_ignore_log = True # change boolean variable

                for game in games_to_play: # "sticking games" that couldn't be played correctly

                    s = game_order[game][0] 
                    t = game_order[game][1] 

                    sticking_matrix[20-s,t] += 1 # add 1 to indicate the games 

        # Check termination condition
        if games_to_play == []: 
            condition = True # all games completed
        else:
            iter_loop += 1 # continue with remaining games

        del llm_short # delete from memory qwen
        gc.collect()

# SAVE MATRICES 
# Convert raw cooperation counts to rates by dividing by repetitions
game_matrix = game_matrix/repetitions
sticking_matrix = sticking_matrix/repetitions

# Save final cooperation matrix to text file
np.savetxt('cooperation1.txt', game_matrix, fmt = '%.2f')
np.savetxt('sticking1.txt', sticking_matrix, fmt = '%.2f')
