# Eye tracking study for TU-Dortmund Case Studies
This project contains a data-pipeline which analyses eyetracking data and can be re-run and tuned.

The goal of this part of the project is, to:
1. Make sure we are all using the same preprocessing, dependencies and utils to do our empirical analysis.
2. Make the robustness analysis the least of a headache, that it can be by automating it end-to-end.
3. Make our results easily reporducible.

## Setup
You are required to install [Anaconda](https://www.anaconda.com/download) and [Git](https://git-scm.com/downloads) on your device before continuing. 

1. Open open a terminal session that supports git
2. Navigate to your desired location where you would like to download the project
3. Run `git clone https://github.com/carterh2/eye_tracking`
4. Open the dedicated prompwindow for anaconda (Anaconda prompt on Windows)
5. Navigate to your source folder of the downloaded project
6. Run `conda env create`
7. Run `conda activate nemo_eyetracking`
8. Run `python main.py` to start the pipeline and follow the console prompts

## Must knows when running the data-pipeline
The **data** in the `data/pre_processed` folder **is not tracked** within the repository. You will have to download the csvs from [here](https://osf.io/sk4fr/). 

If you choose to use the updated dataset (which by the way already includes the old data) make sure to also update the `participant_info.csv` file under `results/`.

**Jupyter Notebooks will not be part of this repository**: Honestly this is to be discussed, but i would like to keep the code as lean as possible. This is piece of infrastructure. Notebooks should be shared in our notion workplace :).

## How to develop the pipeline
I have added two files to the original structure of this project:
1. `./utils/post_processing.py`
    - any processing done to the classified fixations before running the analysis part can be put in here
2. `./utils/analysis.py`
    - If you have ran and tested model fitting and hypothesis testing you can put these procedures as a function into this file

Each of these two files have the 'master-functions' `run_post_processing` and `run_age_and_fixation_duration_analysis`. These functions piece together the procedures you developed.

So: Let's say you have developed a function that tests the relationship between age and fixation duration with a mixed effects model and prints the ouput into the console in a jupyter notebook. You can then follow this workflow:
1. Pull changes from the remote repository, to make sure your version of the project is up to date
2. Insert your function into `utils/analysis.py`
3. Push your changes such that everybody can use your function.
