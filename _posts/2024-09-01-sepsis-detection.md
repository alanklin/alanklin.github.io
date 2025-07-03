---
title: 'Sepsis Detection using Machine Learning Techniques'
date: 2025-07-03
permalink: /posts/2024/09/sepsis/
tags:
  - machine-learning
  - healthcare
---

Setting the Stage
------

For Boston College's 2024 Back to Fall Case Competition, graduate students in the Applied Analytics and Economics were invited to tackle an ongoing problem in the medical sphere: *using machine learning techniques to predict the onset of sepsis after hospital admission.* This competition was funded by RCG Global Services, who sent a panel of professionals to judge the competition and provide us with valuable feedback.

My team for this task consists of [Trevor Petrin](https://www.linkedin.com/in/trevor-petrin/), [Pin Lyu](https://www.linkedin.com/in/pin-lyu-0449b1236/), [Angelo Marinaro](https://www.linkedin.com/in/angelo-marinaro/), and myself.

The Kaggle competition where we submitted our predictions can be found [here](https://www.kaggle.com/competitions/the-nexus-of-sepsis).

The Problem
------ 

Sepsis is a life-threatening condition that spurs from a complication of an infection, where chemicals released in the bloodstream to fight an infection triggers inflammation throughout the body. This can lead to tissue damage, organ failure, and even death if left untreated. Each year in the United States, at least 1.7 million adults develop sepsis, with over 350,000 dying in hospitalization or discharged to hospice. So, our challenge is to predict the onset of sepsis prior to its diagnosis. 

We were provided physiological training data and labels that included hourly measurements for patients. There is about 1.09 million observations associated with 28,235 unique patients and over 3 dozen predictors in the original training set. Let's dive deeper into the variables.

It's divided into three main categories: Vital signs, Laboratory values, and Demographics.

Vital Signs


| Variable Name | Relationship to Sepsis | How | Paper Link |
|---------------|------------------------|-----|------------|
| Heart Rate (beats per minute) | Result | Heart rate elevations are common response to infection | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6102166/" target="_blank">Paper</a>|
| O2Sat (Pulse Oximetry) | Indicator | Low oxygen saturation can indicate impaired immune response, patients may need futher evaluation or treatment | [Paper](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2018.02008/full) |


  * Temperature (in Celcius)
  * Systolic BP (mm Hg)
  * Mean arterial pressure (mm Hg)
  * Diastolic BP (mm Hg)
  * Respiration rate (breaths per minute)
  * End tidal carbon dioxide (mm Hg)

* Laboratory Values
  * Excess Bicarbonate (mmol/L)
  * Bicarbonate (mmol/L)
  * Fraction of inspired oxygen (%)
  * pH
  * Partial pressure of CO2 from arterial blood (mm Hg)
  * Oxygen saturation from arterial blood (%)
  * Aspartate transaminase (IU/L)
  * Blood urea nitrogen (mg/dL)
  * Alkaline phosphatase (IU/L)
  * Calcium (mg/dL)
  * Chloride (mmol/L)
  * Creatinine (mg/dL)
  * Bilirubin direct (mg/dL)
  * Serum glucose (mg/dL)
  * Lactate (mg/dL)
  * Magnesium (mmol/dL0)
  * Phosphate (mg/dL)
  * Potassium (mmol/L)
  * Total bilirubin (mg/dL)
  * Troponin I (ng/mL)
  * Hematocrit (%)
  * Hemoglobin (g/dL)
  * Partial thromboplastin time (seconds)
  * White Blood Cell Count
  * Fibrinogen (mg/dL)
  * Platelet count

  * Demographics