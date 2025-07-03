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

Exploratory Data Analysis
------ 

Defining Variables
====== 

These variables are divided into three main categories: Vital signs, Laboratory values, and Demographics. Each member researched and compiled the following information to help us better understand what each variable is and how it relates to sepsis. We used this information to help us dial in on more promising and available indicators.

* Vital Signs


| Variable Name | Description of Relationship to Sepsis| Information Link |
|---------------|------------------------|------------|
| Heart Rate (beats per minute) | Heart rate elevations are common response to infection | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6102166/" target="_blank">Link</a> |
| O2Sat (Pulse Oximetry) | Low oxygen saturation can indicate impaired immune response, patients may need futher evaluation or treatment | <a href="https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2018.02008/full" target="_blank">Link</a> |
| Temperature (in Celcius) | Infections are associated with rising body temperatures. Some individuals can experience lower body temperature, as well. | <a href="https://www.nidirect.gov.uk/news/recognising-signs-and-symptoms-sepsis#:~:text=The%20early%20symptoms%20of%20sepsis,chills%20and%20shivering" target="_blank">Link</a> |
| Systolic BP (mm Hg)  | Low blood pressure can cause septic shock, but not sepsis itself. Falling below 100 can be indication of sepsis. | <a href="https://www.kidney.org/kidney-topics/sepsis#:~:text=Diagnosis-,Diagnosing%20sepsis%20requires%20a%20medical%20assessment%20by%20a%20healthcare%20professional,than%20or%20equal%20to%20100." target="_blank">Link</a> |
| Mean arterial pressure (mm Hg)  | MAP = DP + (SP-DP)/3.  Can decrease due to vasodilation and cardiac dysfunction | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10492407" target="_blank">Link</a> |
| Diastolic BP (mm Hg)  | Blood pressure still an indicator for sepsis, might actually be more important than systolic! | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10492407" target="_blank">Link</a> | 
| Respiration rate (breaths per minute) | Generally increases during sepsis, patients can have trouble breathing fighting off infection and fever | <a href="https://www.jointcommissionjournal.com/article/S1553-7250(18)30040-0/abstract" target="_blank">Link</a> |
|End tidal carbon dioxide (mm Hg) | Low values have ben used to predict mortality in potentially septic patients | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5942006/" target="_blank">Link</a> |



* Laboratory Values


| Variable Name | Description of Relationship to Sepsis| Information Link |
|---------------|------------------------|------------|
| Excess Bicarbonate (mmol/L) | Relates to how much base is needed to return acific blood levels back to normal. Could be used as predictor of lactate. | <a href="https://pubmed.ncbi.nlm.nih.gov/21159466/" target="_blank">Link</a> |
| Bicarbonate (mmol/L) | Low value is a possible indicator, HCO3 is a base that helps to maintain body's acid-base balance | <a href="https://www.cureus.com/articles/81789-serum-bicarbonate-reconsidering-the-importance-of-a-neglected-biomarker-in-predicting-clinical-outcomes-in-sepsis#!/" target="_blank">Link</a> |
| Fraction of inspired oxygen (%) | Controlled by respiratory tools in hospital. Low FiO2 may be indicator of sepsis, which can cause respiratory problems. | <a href="https://pubmed.ncbi.nlm.nih.gov/23726018/" target="_blank">Link</a> |
| pH  | Lactic acidosis occurs when lactic acid from anoxic tissues overwhelms the blood's buffering capacity. Often indicates severity of sepsis. | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5225760/" target="_blank">Link</a> |
| Partial pressure of CO2 from arterial blood (mm Hg) | Decreased paCO2 is an established clinical indicator for sepsis. Higher values also associated with higher risk of mortality. | <a href="https://journals.lww.com/jtccm/fulltext/2023/06000/paco2_levels_at_admission_influence_the_prognosis.12.aspx" target="_blank">Link</a> |
| Oxygen saturation from arterial blood (%) | Relatively similar to O2Sat but is determined in a more invasive manner | <a href="https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2018.02008/full" target="_blank">Link</a> | 
| Aspartate transaminase (IU/L) | Enzyme released when liver or muscles are damaged, can be indicative of liver damage due to sepsis | <a href="https://www.frontiersin.org/journals/cellular-and-infection-microbiology/articles/10.3389/fcimb.2025.1504223/full" target="_blank">Link</a> |
| Blood urea nitrogen (mg/dL) | Main end product of protein metabolism. Increases during excessive protein breakdown, rate of protein catabolism increases in patients with sepsis | <a href="https://apm.amegroups.org/article/view/82671/html" target="_blank">Link</a> |
| Alkaline phosphatase (IU/L) | Component of host defense against inflammation. Can improve kidney function of patients | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6148963/" target="_blank">Link</a> |
| Calcium (mg/dL) | Crucial to vital physiological processes. U-shaped correlation between serum calcium levels and in-hospital mortality from sepsis. | <a href="https://www.sciencedirect.com/science/article/pii/S2405844024107335" target="_blank">Link</a> |
| Chloride (mmol/L) | Vital for maintenance of acid-base balance, fluid homeostasis, renal function, and more.. | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5899079/ " target="_blank">Link</a> |
| Creatinine (mg/dL)  | Evaluation of renal failure, when it rises, renal function can decrease by 50% | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4894127/" target="_blank">Link</a> |
| Bilirubin direct (mg/dL) | Diagnoses hepatic dysfunction/failure, increased levels are a late event inthe course of multi-organ dysfunction | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3682239/" target="_blank">Link</a> |
| Serum glucose (mg/dL) | Increased glucose production associated with inflammatory mediators and even a pathological acute stress response | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7065801/" target="_blank">Link</a> | 
| Lactate (mg/dL) | Increased levels may represent tissue hypoperfusion associated with signs of organ dysfunction | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4958885/ " target="_blank">Link</a> |
| Magnesium (mmol/dL0) | Loss of magnesium homeostasis exacerbated sepsis progression, hypomagnesemia potential risk factor of infections. | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10304084/" target="_blank">Link</a> |
| Phosphate (mg/dL) | Hypophosphatemia develops in early stages of sepsis, severe cases have a higher mortality rate | <a href="https://pubmed.ncbi.nlm.nih.gov/16501239/" target="_blank">Link</a> |
| Potassium (mmol/L) | Anything that causes many cells to die at once can leak large amounts of potassium into the blood | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11627424/#:~:text=In%20critically%20ill%20patients%20with%20sepsis%2C%20serum%20potassium%20levels%20exceeding,outcomes%20in%20this%20patient%20population." target="_blank">Link</a> |
| Total bilirubin (mg/dL) | Bilirubin increase as a result of body trying to fight bacteria infections | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10259638/" target="_blank">Link</a> |
| Troponin I (ng/mL) | Cardiac troponins are elevated in 85% of patients with sepsis in the absence of acute coronary syndrome | <a href="https://pubmed.ncbi.nlm.nih.gov/19585902/#:~:text=Abstract,did%20not%20improve%20mortality%20rates." target="_blank">Link</a> |
| Hematocrit (%) | Sepsis is characterized by a reduction of hematocrit | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8947025/" target="_blank">Link</a> |
| Hemoglobin (g/dL) | Hemoglobin released from RBC during sepsis, resulting in hemolytic anemia and an increased risk of mortality | <a href="https://www.sciencedirect.com/science/article/abs/pii/S0147956323002832" target="_blank">Link</a> | 
| Partial thromboplastin time (seconds) | When PTT is high, blood is not clotting well. Related to platelet count, could be a sign of sepsis | <a href="https://www.sepsis.org/sepsis-basics/testing-for-sepsis/" target="_blank">Link</a> |
| White Blood Cell Count | WBC at elevated levels are a sign of a current infection, lower levels signifies susceptibility to infection | <a href="https://www.yalemedicine.org/conditions/sepsis#:~:text=Blood%20tests%20may%20reveal%20the%20following%20signs%20suggestive,person%20is%20at%20higher%20risk%20of%20developing%20one." target="_blank">Link</a> |
| Fibrinogen (mg/dL) | Decreased levels occurs only in later stages of sepsis | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9961497/" target="_blank">Link</a> |
| Platelet count | Low platelet count is a key sign of sepsis | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679237/" target="_blank">Link</a> |



  * Demographics


| Variable Name | Description |
|---------------|------------------------|
| Age | in Years (encoded as 100 for patients 90 or above)|
| Sex | Female encoded as 0, Male as 1 |
| Unit1 | Administrative identifier for ICU unit (MICU) |
| Unit2 | Administrative identifier for ICU unit (SICU)|
| HospAdmTime | Hours between hospital admit and ICU admit|
| ICULOS| ICU length-of-stay (hours since ICU admit) | 



