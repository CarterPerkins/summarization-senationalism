# Sensationalism and Gender Bias in News Headlines using Fine-tuned Summarization Models
A study of gender bias and sensationalism on summarization models fine-tuned for headline generation.

---

## Project Description

This project examines the relationship between a news article and its headline by understanding the importance of gender bias and sensational words. This build upon the article *When Women Make Headlines*[^1] where the authors compared the usage of sensational words in headlines about women to headlines about other topics. We extend their analysis by looking further into machine generated headlines and the articles themselves in order to understand potential biases in summarization models.

---

## Repository Structure

This table descibe the repository structure.

| **Directory** | **Description** |
| --- | --- |
| `src/` | Contains source code for fine-tuning summarization models for headline generation, collecting sentiment and polarization data of headlines, and NYU HPC Greene SLURM jobs scripts. |
| `figures/` | Contains figures for slides and observations. |

---

## Experimental Data

To access the experimental data such as the trained models and SLURM logs, visit the [Google Drive](https://drive.google.com/drive/folders/1_SHEAVvvQO5BBgFzADNr7L8VikgxN4J4) folder. Each folder is a specific experiment with the following files:

| **File Extension** | **Description** |
| --- | --- |
| `.params` | Hyperparameters for the model. Columns include: 'epochs', 'model', 'batch_size', 'learning_rate', 'article_max_len', 'headline_max_len', 'split', 'seed', and 'results_name'. |
| `.out` | SLURM stdout log. Files beginning with 'T' are for fine-tuning the summarization model and files beginning with 'E' are for collecting the generated headlines on the test dataset. |
| `.err` | SLURM stderr log. Files beginning with 'T' are for fine-tuning the summarization model and files beginning with 'E' are for collecting the generated headlines on the test dataset. |
| `.test` | CSV file of the same form as the test dataset generated in `src/data` with the following changes: renaming column 'content' to 'article', renaming column 'title' to 'headline', and new column 'generated_headline' from performing inference on 'article' using fine-tuned model. |
| `.pt` | Model weight for summarization model. |
| `.csv` | CSV file of training performance for fine-tuning summarization model. Columns include: 'rougeL', 'rouge1', 'rouge2', 'loss', 'epoch', and 'batch_time'. |

---

## Recreating Experiments

**Dependencies**

Available as `pip` and/or `conda` installations:

* `transformers`
* `datasets`
* `accelerator`
* `rouge-score`
* `pytorch`
* `numpy`
* `pandas`
* `nltk`
* `matplotlib`

NLTK and certain transformer models may require additional installations. Follow error outputs to install.

### Fine-tuning Summarization Models

1. Login to NYU HPC Greene, navigate to your `/scratch` directory, and clone this repository.
2. Install dependencies in a conda environment.
3. Download the dataset using the instructions in `src/data/README.md`.
4. Set working directory to `src/`. Edit the `jobs/summarization/train/runner.sh` file to load the conda environment from (2), and point to the correct path of the cloned repo's `src/` folder.
5. Run `./jobs/summarization/train/batch.sh` to enqueue the SLURM jobs.
6. Generated files will be located in `src/out/`.

### Collecting Generated Headlines

1. Download experimental data (specifically `.pt` and `.params` files) or run the previous fine-tuning steps. Make sure the files are located in `src/out`.
2. Install dependencies in a conda environment.
3. Set working directory to `src/`. Edit the `jobs/summarization/eval/runner.sh` file to load the conda environment from (2), and point to the correct path of the cloned repo's `src/` folder.
4. Run `./jobs/summarization/eval/batch.sh` to enqueue the SLURM jobs.
5. Generated files will be located in `src/out/`.


## Observations


### Summarization Observations

#### Training Loss

![Training Loss](figures/TrainingLoss.svg)

* Doubling input article size of BART models had a negligible effect on loss convergence.
* The models have a similar rate of convegence, but vary in the loss convergence value.

#### Training Batch Time

![Training Batch Time](figures/TrainingBatchTime.svg)

* Doubling input article size lead to a ~1.5x increase for BART and a ~1.8x increase for T5.
* PEGASUS is the slowest model to train, and the previous observation explains why the model did not train in time.

#### Validation Median RougeL

![BART Median RougeL](figures/BARTValidationRougeL.svg)

![T5 Median RougeL](figures/T5ValidationRougeL.svg)

![PEGASUS Median RougeL](figures/PEGASUSValidationRougeL.svg)

* Median RougeL is noisy which can potentially be partially attributed to the small batch size.
* Doubling input article size leads to higher and more consistent RougeL scores.

---

## References
[^1]: https://pudding.cool/2022/02/women-in-headlines/

1. https://github.com/cjhutto/vaderSentiment
3. https://www.kaggle.com/datasets/snapcrack/all-the-news
