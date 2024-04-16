# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire. 

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we? 
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values 
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Plot twist! The expert who priced these gems has now vanished. 
Francesco needs you to be the new diamond evaluator. 
He's looking for a **model that predicts a gem's worth based on its characteristics**. 
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag. 

Create a Jupyter notebook where you develop and evaluate your model.

#### Challenge 2

Good news! Francesco is impressed with the performance of your model. 
Now, he's ready to hire a new expert and expand his diamond database. 

**Develop an automated pipeline** that trains your model with fresh data, 
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

Finally, Francesco wants to bring your brilliance to his business's fingertips. 

**Build a REST API** to integrate your model into a web app, 
making it a cinch for his team to use. 
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

#### Challenge 4

Your model is doing great, and Francesco wants to make even more money.

The next step is exposing the model to other businesses, but this calls for an upgrade in the training and serving infrastructure.
Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨

---

## How to run

### Set up the environment

Python >= 3.10 is required in order to correctly run the code, together with
the dependencies listed in `requirements.txt` that can be installed as follows:

```bash
pip install -r requirements.txt
```

N.B. in order to execute the notebooks and render all the figures, it is also
necessary to install [Graphviz](https://graphviz.org/download/).

#### Anaconda environment

An anaconda virtual environment can be directly created from `env.yaml`.
```bash
conda env create -f env.yaml
conda activate xtream
```

If you want to run the jupyter notebooks it is also necessary to create an ipykernel
for the conda env and activate it in the kernel tab of the notebook (Kernel>Change kernel>xtream).
```bash
python -m ipykernel install --user --name=xtream
```

### Solution structure

#### Challenge 1

The **first challenge** has been addressed in the following two notebooks:
- [data_exploration.ipynb](notebooks/data_exploration.ipynb) containing the exploratory data analysis, the data preprocessing and encoding and some visual insights on the data.
- [model_selection.ipynb](notebooks/model_selection.ipynb) containing the experimentation performed to find a good model for the diamonds' prices prediction, together with reasoning and step-by-step explanations of the why behind every price tag.

#### Challenge 2

The solution to the **second challenge** has been built with python modules under the `src/` folder.
- [dataset.py](src/dataset.py) contains the `DiamondsDataset` class that gives us a structured way to handle the data and performs all the necessary preprocessing and encodings steps.
- [model.py](src/model.py) contains the `DiamondPricePredictor` class which lets you easily train the model with fresh data and make predictions, with the option of generating an explanation for each sample. The class automatically finds the best hyperparameters using grid search. The explanations are wrapped in the `PredictionExplanation` and `DecisionStep` classes.
- [constants.py](src/constants.py) contains some default and constant values that can be overwritten to reflect on the whole pipeline.

An example of running the whole training pipeline is show below:
```py
from src.dataset import DiamondsDataset
from src.model import DiamondPricePredictor

# we can give as many csv files as we want to build the training dataset
dataset = DiamondsDataset.from_csv('datasets/diamonds/diamonds.csv',
                                   'datasets/diamonds/fresh_diamonds_data.csv')

# the best model configuration is found with grid search
model = DiamondPricePredictor.from_grid_search_cv(dataset)
```

If we want keep a portion of the data for testing purposes and predict prices with our model,
together with explanations of the why behind every price tag, we can do as follow:
```py
from src.dataset import DiamondsDataset, load_csv_data
from src.model import DiamondPricePredictor

train_set, test_set = DiamondsDataset.train_test_split(load_csv_data('datasets/diamonds/diamonds.csv'))

model = DiamondPricePredictor.from_grid_search_cv(train_set)

predictions = model.predict(test_set)
explanations = model.predict_explain(test_set)
metrics = model.evaluate(test_set, predictions)
```
#### Challenge 3

For the **third challenge**, a small Flask app has been built in [app.py](src/app.py) to make the whole pipeline available as REST APIs.
It can be run with:

```bash
flask --app src.app run
```

##### Predict Price
Endpoint: `/predict`

Method: `GET`

Request Body: JSON list of diamond characteristics (in the format of DiamondSample dataclass instances).

Returns: Predicted prices for the input samples.

Return Type: JSON list of floats.

##### Predict Price with Explanation
Endpoint: `/predict-explain`

Method: `GET`

Request Body: JSON list of diamond characteristics (similar to the `/predict` endpoint).

Returns: Predicted prices and decision steps for the input samples.

Return Type: JSON list of objects, where each object contains the predicted price and the decision steps involved in the prediction.

##### Re-train the Model with specific datasets
Endpoint: `/train`

Method: `POST`

Request Body: JSON list of strings representing the dataset filenames to be used for training.

Returns: Model evaluation metrics.

Return Type: JSON object with evaluation metrics.

##### List available Datasets
Endpoint: `/datasets`

Method: `GET`

Returns: A list of available dataset filenames within the datasets directory.

Return Type: JSON list of strings

##### Manage individual Dataset
Endpoint: `/datasets/<dataset_name>`

Methods: `GET`, `PUT`, `DELETE`

Parameters: dataset_name (URL parameter): Name of the dataset file.

Request Body:
- PUT: JSON list of diamond characteristics (in the format of DiamondSample dataclass instances) to create the dataset with.

Returns:
- GET: The contents of the specified dataset.
- PUT: Confirmation of dataset creation or update.
- DELETE: Confirmation of dataset deletion.

Return Type:
- GET: JSON list of dictionaries representing the dataset rows.
- PUT/DELETE: String message.

##### Notes
All endpoints return "Invalid method" if an unsupported HTTP method is used.

The `/datasets/<dataset_name>` `PUT` method and both predict endpoints (`/predict` and `/predict-explain`) expect a request body in JSON format. Incorrect or incomplete data structures result in an error message and a 400 Bad Request status code.

The `/train` endpoint requires the request body to be a list of dataset names available within the specified datasets directory. Absence of any listed dataset in the directory will result in an error message and a 400 Bad Request status code.

