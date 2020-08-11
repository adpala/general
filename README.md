# How does a fly integrate information from its group during social interactions?

**Context:** Clean (high confidence of tracking) dynamic group behavior (chaining), controlling social environment (amount of flies in the vecinity).

## General / Cues

* [General, Cues] Can we predict the movement of a fly within the group behavior?
    *   Performance of models for velocity.
    *   Performance of models for acceleration.
    *   Performance of models for rotation.
    *   Performance of models for different position in chain.
    *   Performance of models for different "social environments".

## Attention

* [Attention] What mutations can be relevant to test attention (visual?) of flies during group behavior?

* [Attention, Cues] How do GLM models distinguish between target flies and their features?
    *   Explain the characteristics and differences of the provided filters.

* [Attention, Cues] What are the most relevant sources of information for the models?
    *   Front flies over back flies.
    *   Potential contribution of not-closest flies.
    *   Is nearest-frontal-neighbor our best predictor?

* [Attention, Cues] Do flies attend to individual flies or they average them, as if responding to the optic flow?

* [Attention] How is general chaining activity affected by alteration of visual attention (through sleep deprivation, following *Kirszenblat 2018*)?

* [Attention] Can we reproduce the effect of distractors in NM91 or OregonR in a fly-ball setting, similar to those in *Frighetto 2019* (small deviation from ongoing targeted motion)?
    *   It requires:
        * [ ] Closed-loop visual input, to adjust size and location of targets as fly moves towards them.

## Fly-ball

* Can we reproduce chaining behavior on a ball and virtual visual reality?
* How does a fly respond to (similar to *Kirszenblat 2018*):
    *   static distractors?
    *   moving distractors?
    *   gradually growing or shrinking targets?

## Target Selection

**Context:** Particularly unstable chains or wing-choice.

* [Target-Selection] Can we automatically detect change of target choice events? For either chaining or wing-extension.

* [Target-Selection] What triggers a fly to change its target choice? Followed fly in chaining, change of wing in wing extension.
    *   Do flies change choice randomly or is there a common pattern? a change in position of their target?
    *   Does a fly change its target based on visual cues? Sudden growth/motion of a big proportion of its visual field?
        *   What is the common visual field of a fly during change of target?
        *   What is the average change of the visual field?
        *   Is there a correlation on the chosen target and the change of the visual field?
        *   Activate or inhibit E-PG neurons that correspond to a specific side, and measure the effect on target selection or distractors (like *Fisher 2019*).
    *   Calculate a Wing Choice Change Average, WCCA. Or a Target Choice Change Average.
    *   Provoke change of target artificially (fly-ball). Requires:
        * [ ] fly ball setup

* [Target-Selection] How often does a fly change its target choice?
    *   Can we use this change of target as a readout or measure relevant to study the brain?
    *   Calculate proportion of bouts with mixed events. Requires:
        * [ ] Bout data collection

* [Target-Selection] In wing extension, how fast does the predicted fly change its wing to optimize the antenal reception of sound (in normal courtship or not crowded situations)?
    *   Is it even optimizing the sound reception to the target fly in the male-male courtship, or it optimizes it to the average of its visual input?
    *   Requires:
        * [ ] Sophisticated quantification of relative orientation and occlusion (not only for different flies, but for body parts head-thorax-tail)

## Epochs

* [Epochs] Are there stereotypical epochs of locomotion or behavior that can be used to divide our data into epochs?
    *   Do particular models (individual models or generalized/averaged models) perform better than others at certain epochs?
    *   Do models perform in different way at characteristic locomotion epochs (acceleration, deceleration, stop).

## Spread

* [Spread] How does nearby chaining activity spread to other individuals?
    *   Track history of a chain. How does it start? How does it grow? How does it end?
    *   How does the size of chains change the probability of growing the chain?
    
* [Spread] Can we explain spread of chaining behavior with a contagion model?
    *   Dorothea's project

* [Spread] How are wing extensions spreading through the individuals in the chamber or in the chain?

* [Spread] Do flies behave synchronized, alternated or stochastically?

## Purpose

* [Purpose] What is the purpose of chains?
    *   Cooperation to improve copulation chances?
        *   Do they chain when females are present?
        *   Do they shorten copulation latency when the group is chaining more near a female?
    *   Confusion?
        *   Do they stop chaining when females are present? How many females are required to stop all chaining behavior?
        *   How much do they chain as a function of distance to the females?
    *   Practice?
        *   Do males that chain a lot in a video have better success rates? (new chamber where female is not available until the end).


