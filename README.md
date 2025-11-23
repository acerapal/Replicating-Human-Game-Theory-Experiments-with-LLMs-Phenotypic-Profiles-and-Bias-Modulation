# Replicating-Human-Game-Theory-Experiments-with-LLMs-Phenotypic-Profiles-and-Bias-Modulation
Code used for my master thesis. The code can be used to play dyadic games with enhanced prompting techniques, including multi-step reasoning and logical verification, to generate cooperation matrices. In each code we have tested different LLMs: Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B, to which we refer as Llama, Qwen and Mistral for simplicity.

| File        | Description |
|---------------|---------------------|
| data   | Matrices needed to recreate the plots                  |
| Llama_finalversion.py   |  Play games with Llama                   |
| Mistral_finalversion.py   |    Play games with Mistral                 |
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




