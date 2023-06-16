# Artificial Artificial Artificial Intelligence (In Progress)

This repository hosts the work in progress related to our paper `Artificial Artificial Artificial Intelligence: Crowd Workers Widely Use Large Language Models for Text Production Tasks`

## MTurk Responses
The original responses and abstracts are made available at `summaries.csv` and `abstracts_final.csv`. Additionally, the code to re-create the human intellgience task (HIT) is at `hit.html`. The processed data, new batches, and predictions are left in `data/`. 

## Repository Structure
Here's a breakdown of the notebooks included in the repository and the purpose of each:

1. `0_exploration.ipynb` - Processing the original responses.
2. `1_generation.ipynb` - Generating new responses using ChatGPT.
3. `2_prepare_for_training.ipynb` - Preparing the training data.
4. `3_finetuned_classifying.ipynb` - Plotting and detection post-training.

## Fine-Tuning the Model
The process of fine-tuning the model was largely based on the code repository found [here](https://github.com/AGMoller/worker_vs_gpt/). We are in the process of integrating the relevant fine-tuning code into this repository. 

_Note: Information about the training of the model will be added once the process is completed._

## Citation

```bibtex
@misc{veselovsky2023artificial,
      title={Artificial Artificial Artificial Intelligence: Crowd Workers Widely Use Large Language Models for Text Production Tasks}, 
      author={Veniamin Veselovsky and Manoel Horta Ribeiro and Robert West},
      year={2023},
      eprint={2306.07899},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}