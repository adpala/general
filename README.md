# Project Structure

## General information

**Main Question:** How do flies integrate (social) information in groups (chains)?

**Context:** Clean (high confidence of tracking) dynamic group behavior (chaining), controlling social environment (amount of flies in the vecinity).

## Cues and Sources

**Question:** What do flies care the most during group behavior (chaining)?

 Which model is the simplest effective way to predict the behavior of a fly in a group?
* Implement all methods in the wing choice analysis, by adjusting which data the algorithms receives or how much does it weight it depending on the model. Does the wing choice paradigm break at any of the models? Which is the simplest of the models that still follows the paradigm?

How do GLM models distinguish between the flies in the surrounding? Do they prefer (higher filter coefficients) any fly in particular (e.g. closest fly or front flies)?
* Train GLM models with all available data for the flies, and compare filters. Which are the better filters? How different/unique are the filters?

## Epochs

**Question:** When does the self model fail? Do social models have a better prediction here?

* Train a model with only self information, and compare predictions and accuracy to other models. When does it fail? Do social models improve the prediction? What is characteristic of these moments that would require social information?

## Timing

**Context:** Target choice change (either wing extension or chaining)

* Define event detection

**Question:** What is the time integration delay of flies within group behavior?

How often do flies change choice?

* Calculate proportion of bouts with mixed events. How often do they change choice?

Do flies change choice randomly or because of a change in position of their target?

* Calculate a Wing Choice Change Average, WCCA. Or a Target Choice Change Average.

How long does it take to flies to change their wing choice after the target has moved from one side to the other?

* Compare time delay in the WCCA. Is this consistent across events?

Which body part of the fly more likely to be followed by this behavior (head, thorax or tail)?

Do they follow the average visual field, following the group instead of a single fly?


## Spread

**Question:** How is chaining-drive spread through the individuals?

How does the size of chains change the probability of growing the chain?

* Track history of a chain.

Which contagion model explains the best how flies influence each other?

* Dorothea's project

How are wing extensions spreading through the individuals in a chain? How do they differ from not-chaining flies?

* Do flies behave synchronized, alternated or stochastically?
* Which contagion model explains the best how flies influence each other?

## Purpose

**Question:** What is the purpose of chains?

Cooperation to improve copulation chances?
* Do they chain when females are present?
* Do they shorten copulation latency when the group is chaining more near a female?

Confusion?
* Do they stop chaining when females are present? How many females are required to stop all chaining behavior?
* How much do they chain as a function of distance to the females?

Practice?
* Do males that chain a lot in a video have better success rates? (new chamber where female is not available until the end).


