# Replicating-Human-Game-Theory-Experiments-with-LLMs-Phenotypic-Profiles-and-Bias-Modulation
The code can be used to play dyadic games with enhanced prompting techniques, including multi-step reasoning and logical verification, to generate cooperation matrices. We have tested different mid-sized LLMs: Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B, to which we refer as Llama, Qwen and Mistral for simplicity.

| File        | Description |
|---------------|---------------------|
| data   | Matrices needed to recreate the plots                  |
| Llama_Mistral_finalversion.py   |  Play games with Llama or Mistral                  |
| Qwen_finalversion.py  |  Play games with Qwen                |
| Plots.ipynb   |      Code for the plots               |
| Nash.ipynb   |     Code for getting Nash equilibrium cooperation                |

## Data Files

The `data/` folder contains the following matrices:

| File | Description |
|------|-------------|
| cooperation1.txt | Average Llama cooperation over 10 iterations for new and old games|
| copperation2.txt |Average Llama cooperation over 10 iterations for new and old games|
| llama_extract.txt | Average Llama cooperation over 20 iterations using double extraction |
| llama_multi.txt| Average Llama cooperation over 20 iterations using double extraction and logical steps |
| llama_simple.txt | Average Llama cooperation over 20 iterations using simple extraction|
| llama10rep_final1.txt | Average Llama cooperation over 10 iterations using double extraction, logical steps and logical verifier|
| llama10rep_final2.txt | Average Llama cooperation over 10 iterations using double extraction, logical steps and logical verifier|
| matrix_human.txt| Average human cooperation provided by the original experiment Poncela-Casasnovas et al. (2016)|
| mistral_extract.txt | Average Mistral cooperation over 20 iterations using double extraction |
| mistral_multi.txt| Average Mistral cooperation over 20 iterations using double extraction and logical steps |
| mistral_simple.txt | Average Mistral cooperation over 20 iterations using simple extraction|
| mistral10rep_final1.txt | Average Mistral cooperation over 10 iterations using double extraction, logical steps and logical verifier|
| mistral10rep_final2.txt | Average Mistral cooperation over 10 iterations using double extraction, logical steps and logical verifier|
| matrix_nash.txt | Nash cooperation|
| newnash.txt | Nash cooperation for old and new games|
| qwen_extract.txt | Average Qwen cooperation over 20 iterations using double extraction |
| qwen_multi.txt| Average Qwen cooperation over 20 iterations using double extraction and logical steps |
| qwen_simple.txt | Average Qwen cooperation over 20 iterations using simple extraction|
| qwen10rep_final1.txt | Average Qwen cooperation over 10 iterations using double extraction, logical steps and logical verifier|
| qwen10rep_final2.txt | Average Qwen cooperation over 10 iterations using double extraction, logical steps and logical verifier|
| sticking1.txt | Proportion, over 10 iterations, Llama couldn't find an answer for the game that passed the logical verifier|
| sticking2.txt | Proportion, over 10 iterations, Llama couldn't find an answer for the game that passed the logical verifier|

For clarification, if it's not indicated in the description which games were simulated (old or new), it means only the old games were played. The matrices that have the same name but end with 1 or 2, were simualted with the same code but the number of iterations is only 10 due its high computational cost. They are averaged to get the final matrix used for the plots.

## Code for the Simulation

`Llama_Mistral_finalversion.py` and `Qwen_finalversion.py` share the same core logic. The main differences are:
- The function that adapts prompts to each LLM's specific format
- Model loading paths
- Model management: `Llama_Mistral_finalversion.py` deletes and reloads models between steps to manage memory, while `Qwen_finalversion.py` uses the same loaded model throughout

These scripts can be adapted to play different game combinations and test other models. Below are the key parameters to modify for your own simulations:

### Key Variables

| Variable | Description |
|----------|-------------|
| model_path_*  | Path to your LLM |
| params_explain | Temperature and maximum tokens of model that generates the long answer |
| params_extract | Temperature and maximum tokens of model that extracts short answer and acts as logical verifier |
| S | Array that contains the Sucker's Payoff of the games to play |
| T | Array that contains the Temptation's Payoff of the games to play |
| repetitions | Number of iterations |
| llm_long | model that generates long answer |
| llm_short | model that extracts short answer and acts as logical verifier |

The code `Llama_Mistral_finalversion.py` is adapted for both LLMs. To switch from one model to the other:
- When defining llm_long, use the right path
- When generating the prompt, change the function that adapts its structure (convert_messages_to_prompt_*)
- Remember to change, if wanted, the names of the resulting .txt files

### Prompt Variables

| Variable | Description |
|----------|-------------|
| instructions_script_short | Prompt that contains the instructions for the games and the prize (euros per point)|
|all_values_script_0| Prompt that contains the possible outcomes for all games when A is associated with cooperation and B to defection|
|all_values_script_1| Prompt that contains the possible outcomes for all games when B is associated with cooperation and A to defection|
|steps| Prompt that contains the logical steps to help the LLM reach a reasonable result|
|message| Prompt that containts the instructions for the logical verifier, what is considered acceptable and what is not, with real exemples|

It's important to notice that if you want to run a model other than Mistral, Qwen or Llama, the prompt needs to be adapted to this new format. 




