[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/twilJ7f4)
# Regression

- Repository: `challenge-regression`
- Type of Challenge: `Consolidation`
- Duration: `5 days`
- Deadline: `06/05/2025 17:00`
- Solo Challenge

## Learning objectives

- Be able to preprocess data for machine learning.
- Be able to apply a regression in a real context.
- Be able to understand some of  machine learning.


## The Mission

The real estate company "ImmoEliza" asks you to create a machine learning model to predict prices on Belgium's real estate sales.

You have **collected** your data, you have **cleaned** and **analyzed** it a first time!
So it's time to do some machine learning with it!

### Must-have features

#### Step 1 : Data cleaning

Preprocess the data to be used with machine learning.

- You have to handle NANs.
- You have to handle categorical data.
- You have to select features.

#### Step 2: Data split

Now that the dataset is ready, you have to format it for machine learning:

- Divide your dataset for training and testing. (`X_train, y_train, X_test, y_test`)

#### Step 3: Model selection

The dataset is ready. Now let's select a model.

Look at which models make the most sense according to your data.

#### Step 4: Apply your model

Apply your model on your data:

- Train your model (on the train dataset)
- Check for predictions (on single lines or the test dataset)
- Once this works, look into ``sklearn``'s ``Pipeline`` object to make things clean and reusable

#### Step 5: Model evaluation

Let's evaluate your model. The metric we are interested in is the MAE (Mean Absolute Error). Make sure you understand it well. Try to answer those questions:

- How could you improve this result?
- Which part of the process has the most impact on the results?
- Are there other metrics which would make more sense to evaluate your model.

You may go back a couple of steps if you want to try other types of approaches.

#### Bonus Step 5.5: Reinventing the wheel

I know some of you will get to a viable model really quickly and will get bored to go back and forth between filtering out outliers and selecting features. The truth is when playing with ML, you only truly understand it when you do it yourself. Here is what you can do:

- Watch what most ML models do to make a prediction
- Select one which you find elegant
- Implement it from scratch using at maximum ``numpy``

Note that some are easier to implement than others.

#### Step 6: Presentation

Present your results in front of the group.

- You have to make a nice presentation **with a professional design**.
- You have **5 minutes** to present (without Q&A). **You can't use more time**, **you can't use less time**.
- You **CAN'T show code or jupyter notebook** during the presentation.

## Constraints

### Code style

- Each **function or class** has to be **typed**
- Each **function or class** has to contain a **docstring**
- Your code should be **commented** when necessary.
- Your code should be **cleaned of any unused code**.

## Deliverables

1. Pimp up the README file:
   - Description
   - Installation
   - Usage
   - (Visuals)
   - (Contributors)
   - (Timeline)
   - (Personal situation)
2. Present your results in front of the group in **5mins max**.

### Steps

1. Create the repository
2. Study the request (What & Why ?)
3. Identify technical challenges (How ?)

## Evaluation criteria

| Criteria       | Indicator                                     | Yes/No |
| -------------- | --------------------------------------------- | ------ |
| 1. Is complete | Know how to answer all the above questions.   | [ ]    |
|                | `pandas` and `matplotlib`/`seaborn` are used. | [ ]    |
|                | All the above steps were followed.            | [ ]    |
|                | A nice README is available.                   | [ ]    |
|                | Your model is able to predict something.      | [ ]    |
| 2. Is good     | You used typing and docstring.                | [ ]    |
|                | Your code is formatted (PEP8 compliant).      | [ ]    |
|                | No unused file/code is present.               | [ ]    |

## Quotes

“The lottery is a tax on people who don't understand the statistics.”
_- Anonymous_

![You've got this!](https://media.giphy.com/media/5wWf7GMbT1ZUGTDdTqM/giphy.gif)
