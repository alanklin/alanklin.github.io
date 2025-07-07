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

This is a visualization of the general data pipeline we will be following. 

<img src='/images/sepsis-pipeline-picture.png'>



Exploratory Data Analysis
------ 

### Defining Variables
We were provided physiological training data and labels that included hourly measurements for patients. There is about 1.09 million observations associated with 28,235 unique patients and over 3 dozen predictors in the original training set. Let's dive deeper into the variables.


These variables are divided into three main categories: Vital signs, Laboratory values, and Demographics. Each member researched and compiled the following information to help us better understand what each variable is and how it relates to sepsis. We used this information to help us dial in on more promising and available indicators.

* Vital Signs


| Variable Name | Description of Relationship to Sepsis| Information Link |
|---------------|------------------------|------------|
| Heart Rate (beats per minute) | Heart rate elevations are common response to infection | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6102166/" target="_blank">Rudiger, et al. </a> |
| O2Sat (Pulse Oximetry) | Low oxygen saturation can indicate impaired immune response, patients may need futher evaluation or treatment | <a href="https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2018.02008/full" target="_blank">Avendaño-Ortiz, et al.</a> |
| Temperature (in Celcius) | Infections are associated with rising body temperatures. Some individuals can experience lower body temperature, as well. | <a href="https://www.nidirect.gov.uk/news/recognising-signs-and-symptoms-sepsis#:~:text=The%20early%20symptoms%20of%20sepsis,chills%20and%20shivering" target="_blank">NI Direct Govt. Services</a> |
| Systolic BP (mm Hg)  | Low blood pressure can cause septic shock, but not sepsis itself. Falling below 100 can be indication of sepsis. | <a href="https://www.kidney.org/kidney-topics/sepsis#:~:text=Diagnosis-,Diagnosing%20sepsis%20requires%20a%20medical%20assessment%20by%20a%20healthcare%20professional,than%20or%20equal%20to%20100." target="_blank">National Kidney Foundation</a> |
| Mean arterial pressure (mm Hg)  | MAP = DP + (SP-DP)/3.  Can decrease due to vasodilation and cardiac dysfunction | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10492407" target="_blank">Gao, et al.</a> |
| Diastolic BP (mm Hg)  | Blood pressure still an indicator for sepsis, might actually be more important than systolic! | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10492407" target="_blank">Gao, et al.</a> | 
| Respiration rate (breaths per minute) | Generally increases during sepsis, patients can have trouble breathing fighting off infection and fever | <a href="https://www.jointcommissionjournal.com/article/S1553-7250(18)30040-0/abstract" target="_blank">Loughlin, et al.</a> |
|End tidal carbon dioxide (mm Hg) | Low values have ben used to predict mortality in potentially septic patients | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5942006/" target="_blank">Hunter, et al.</a> |



* Laboratory Values


| Variable Name | Description of Relationship to Sepsis| Information Link |
|---------------|------------------------|------------|
| Excess Bicarbonate (mmol/L) | Relates to how much base is needed to return acific blood levels back to normal. Could be used as predictor of lactate. | <a href="https://pubmed.ncbi.nlm.nih.gov/21159466/" target="_blank">Montassier, et al.</a> |
| Bicarbonate (mmol/L) | Low value is a possible indicator, HCO3 is a base that helps to maintain body's acid-base balance | <a href="https://www.cureus.com/articles/81789-serum-bicarbonate-reconsidering-the-importance-of-a-neglected-biomarker-in-predicting-clinical-outcomes-in-sepsis#!/" target="_blank">Paudel, et al.</a> |
| Fraction of inspired oxygen (%) | Controlled by respiratory tools in hospital. Low FiO2 may be indicator of sepsis, which can cause respiratory problems. | <a href="https://pubmed.ncbi.nlm.nih.gov/23726018/" target="_blank">Neto, et al.</a> |
| pH  | Lactic acidosis occurs when lactic acid from anoxic tissues overwhelms the blood's buffering capacity. Often indicates severity of sepsis. | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5225760/" target="_blank">Ganesh, et al.</a> |
| Partial pressure of CO2 from arterial blood (mm Hg) | Decreased paCO2 is an established clinical indicator for sepsis. Higher values also associated with higher risk of mortality. | <a href="https://journals.lww.com/jtccm/fulltext/2023/06000/paco2_levels_at_admission_influence_the_prognosis.12.aspx" target="_blank">Qu, et al.</a> |
| Oxygen saturation from arterial blood (%) | Relatively similar to O2Sat but is determined in a more invasive manner | <a href="https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2018.02008/full" target="_blank">Avendaño-Ortiz, et al.</a> | 
| Aspartate transaminase (IU/L) | Enzyme released when liver or muscles are damaged, can be indicative of liver damage due to sepsis | <a href="https://www.frontiersin.org/journals/cellular-and-infection-microbiology/articles/10.3389/fcimb.2025.1504223/full" target="_blank">Guo, et al.</a> |
| Blood urea nitrogen (mg/dL) | Main end product of protein metabolism. Increases during excessive protein breakdown, rate of protein catabolism increases in patients with sepsis | <a href="https://apm.amegroups.org/article/view/82671/html" target="_blank">Li, et al.</a> |
| Alkaline phosphatase (IU/L) | Component of host defense against inflammation. Can improve kidney function of patients | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6148963/" target="_blank">Don Baek, et al.</a> |
| Calcium (mg/dL) | Crucial to vital physiological processes. U-shaped correlation between serum calcium levels and in-hospital mortality from sepsis. | <a href="https://www.sciencedirect.com/science/article/pii/S2405844024107335" target="_blank">Wang, et al</a> |
| Chloride (mmol/L) | Vital for maintenance of acid-base balance, fluid homeostasis, renal function, and more.. | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5899079/ " target="_blank">Pfortmueller, et al.</a> |
| Creatinine (mg/dL)  | Evaluation of renal failure, when it rises, renal function can decrease by 50% | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4894127/" target="_blank">Bilgili, et al.</a> |
| Bilirubin direct (mg/dL) | Diagnoses hepatic dysfunction/failure, increased levels are a late event inthe course of multi-organ dysfunction | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3682239/" target="_blank">Nesseler, et al.</a> |
| Serum glucose (mg/dL) | Increased glucose production associated with inflammatory mediators and even a pathological acute stress response | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7065801/" target="_blank">Kushimoto, et al.</a> | 
| Lactate (mg/dL) | Increased levels may represent tissue hypoperfusion associated with signs of organ dysfunction | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4958885/ " target="_blank">Lee, et al.</a> |
| Magnesium (mmol/dL0) | Loss of magnesium homeostasis exacerbated sepsis progression, hypomagnesemia potential risk factor of infections. | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10304084/" target="_blank">Saglietti, et al.</a> |
| Phosphate (mg/dL) | Hypophosphatemia develops in early stages of sepsis, severe cases have a higher mortality rate | <a href="https://pubmed.ncbi.nlm.nih.gov/16501239/" target="_blank">Shor, et al.</a> |
| Potassium (mmol/L) | Anything that causes many cells to die at once can leak large amounts of potassium into the blood | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11627424/#:~:text=In%20critically%20ill%20patients%20with%20sepsis%2C%20serum%20potassium%20levels%20exceeding,outcomes%20in%20this%20patient%20population." target="_blank">Zhao, et al.</a> |
| Total bilirubin (mg/dL) | Bilirubin increase as a result of body trying to fight bacteria infections | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10259638/" target="_blank">Shah, et al.</a> |
| Troponin I (ng/mL) | Cardiac troponins are elevated in 85% of patients with sepsis in the absence of acute coronary syndrome | <a href="https://pubmed.ncbi.nlm.nih.gov/19585902/#:~:text=Abstract,did%20not%20improve%20mortality%20rates." target="_blank">Smith, et al.</a> |
| Hematocrit (%) | Sepsis is characterized by a reduction of hematocrit | <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8947025/" target="_blank">Luo, et al.</a> |
| Hemoglobin (g/dL) | Hemoglobin released from RBC during sepsis, resulting in hemolytic anemia and an increased risk of mortality | <a href="https://www.sciencedirect.com/science/article/abs/pii/S0147956323002832" target="_blank">Zhu, et al.</a> | 
| Partial thromboplastin time (seconds) | When PTT is high, blood is not clotting well. Related to platelet count, could be a sign of sepsis | <a href="https://www.sepsis.org/sepsis-basics/testing-for-sepsis/" target="_blank">Sepsis Alliance</a> |
| White Blood Cell Count | WBC at elevated levels are a sign of a current infection, lower levels signifies susceptibility to infection | <a href="https://www.yalemedicine.org/conditions/sepsis#:~:text=Blood%20tests%20may%20reveal%20the%20following%20signs%20suggestive,person%20is%20at%20higher%20risk%20of%20developing%20one." target="_blank">Yale Medicine</a> |
| Fibrinogen (mg/dL) | Decreased levels occurs only in later stages of sepsis | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9961497/" target="_blank">Tsantes, et al.</a> |
| Platelet count | Low platelet count is a key sign of sepsis | <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679237/" target="_blank">Vardon-Bounes, et al.</a>|



  * Demographics


| Variable Name | Description |
|---------------|------------------------|
| Age | in Years (encoded as 100 for patients 90 or above)|
| Sex | Female encoded as 0, Male as 1 |
| Unit1 | Administrative identifier for ICU unit (MICU) |
| Unit2 | Administrative identifier for ICU unit (SICU)|
| HospAdmTime | Hours between hospital admit and ICU admit|
| ICULOS| ICU length-of-stay (hours since ICU admit) | 


### Preparing the Data

Through our research, we gained valuable information about what sepsis is and how some of the indicators could be used in our models. But, we quickly noticed that a number of variables were missing values, particularly the laboratory values. We attributed this to the possibility that testing and taking these values at each hour for each patient can be unfeasible for hospitals to do. We decided that variables with more than 95% missing values were too sparse and removed them from the next step: data imputation.

Data imputation is the process of replacing missing data with substituted values. It enables us to use machine learning models that require complete datasets and helps to retain valuable information instead of discarding incomplete data. 

We settled with a forward/backwards imputation technique which can be demonstrated in the photo below.


<img src='/images/sepsis-imputation.png'>


Forward imputation is as the name suggests, we will fill missing values by taking the last known value forward. So, you can see that the empty red values have been substituted with the preceding value. Backwards fill is simply the opposite, as shown by the blue values. This was a pretty naive and straightforward imputation method, but it didn't always cover all the bases, so we also leaned on MissForest imputation, a technique that utilizes random forests to predict and fill in the remaining missing values. 


### Train/Validation Split

Every machine learning project is incomplete without defining how the data will be split into a training and validation set. We were already provided a test set to generate predictions but we can only evaluate those results by uploading a submission into Kaggle. A validation set created from the training set would help us simulate results and tweak hyperparameters without using up our limit of 5 submissions a day. 

So, in the interest of creating a fair validation set, we performed a stratified sampling method, where we based the split on whether a patient ever developed sepsis at any point during their hospital stay. Essentially, we couldn't just randomize the sampling process because we were dealing with a panel dataset, where we're tracking the same variables for patients over time. If we had simply randomized the rows into a training and validation set, we'd lose the temporal aspect of the study, a key part of the task. Our validation set took 20% of the original training set but we preserved the starting proportions of Sepsis/non-Sepsis patients.

### Feature Engineering

Even with over three dozen potential indicators of sepsis, through our research, we quickly came across other indicators that weren't in the dataset but could be easily created through feature engineering. For example, we can use the patient's respiratory rate to check for Tachypnea, the medical term for rapid and shallow breathing, generally defined as over 20 breaths per minute. Other features that were created includes but aren't limited to the following:

  * **BUN to Creatinine ratio**
  * **Shock Index**
  * **Pulse Pressure**
  * **Partial SOFA score**

I want to direct focus onto the SOFA score, which is also known as the [Sequential Organ Failure Assessment Score](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score). This calculator helps to determine the extent of organ dysfunction in critically ill patients. While it isn't designed specifically to diagnose sepsis because patients will manifest symptoms in unique ways, these parameters can help clinicians identify and react to sepsis.  

We called our variable the partial SOFA score because we didn't have all of the variables to generate the complete SOFA score, but it was still an invaluable feature to include. The calculator linked above shows how different levels of each feature corresponds to a higher SOFA score and we utilized this knowledge in our feature engineering step. 


### SMOTE 

One of the biggest challenges we overcame in this competition was the substantial data imbalance between sepsis/non-sepsis records. We decided to address this by utilizing [SMOTE](https://www.jair.org/index.php/jair/article/view/10302) (Synthetic Minority Oversampling Technique), an oversampling technique where synthetic samples are generated for the minority class. By using imbalanced-learn (a package under sklearn) and it's implementation of SMOTE, we inflated the proportion of sepsis cases to 30% as an attempt to give the algorithm more examples to learn about the variables associated with sepsis diagnosis. 


### Scaling

We fit a Standard Scaler on continuous variables to standardize the range of independent features and ensure that each variable will contribute equally to the model's learning process. We don't want features with larger values to overshadow the others just because it has larger nominal values. In addition, scaling helps improve model convergence and training speed as it requires less iterations to calculate smaller values.

### Drop Variables

The final step in our Data Processing pipeline is to drop variables that won't be used in the model training process. Other than dropping features that were too sparse in the data, we also dropped features that went into calculating the newly engineered features, as we figured the information captured in the new features need not be duplicated by also including the original features. We also moved to drop highly correlated variables to continue simplifyng the model and reduce computation time. By the end of this process, we were working with only 15 features, much less than the initial 42. 


Models
------

We decided to pursue 5 different modeling approaches. Here were our results.

<img src='/images/sepsis-model-results.png'>

Our best performing model by metrics alone is the Neural Network, which I poured countless hours of tuning into, even though the following computational graph for our Neural Network might look quite simple. 

### Model Architecture

<img src='/images/sepsis-nn-architecture.png'>

There are 5 hidden layers that utilize a ReLU activation function as well as batch normalization and dropout layers to aid in training speed, convergence, and model generalization on unseen data. The output layer utilizes a sigmoid activation function to generate a value betweeen 0 and 1 to signify the probability of being a positive sample. By using one neuron in this layer, we ensure that we generate one prediction for each time point. 

### Model Parameters

In addition, the final hyperparameter values after tuning are laid out below.

<img src='/images/sepsis-nn-parameters.png'>

The epoch value is a bit arbitrary as I utilized an early stopping function that tracked validation loss, where it terminated training if 5 epochs had passed without a further decrease in loss. We elected to restore the model weights from the epoch with the the minimum loss value because it demonstrates the best validation performance and represents a model that presumably is not overfit on training data and will generalize well to the unseen test set.

Dropout layers were included to improve model generalization and training speed. By randomly turning off a set of neuron during the training process, we could force the model to rely on other neurons to update its weights. In theory, this would enable the model to make full use of the entire connections within the model architecture, instead of over-relying on a subset of neurons for its predictions. 

The number of hidden layers and neurons was settled upon by striking a balance between training speed and model complexity. I found that a deeper neural network model with more layers but less neurons performed better than a shallow NN model with more neurons. 

### Model Evaluation

<img src='/images/sepsis-nn-parameters.png'>

With these parameters, our best performing neural network achieved an F1-score of 0.1825 on the validation set. At a glance, our model was incredibly successful at predicting non-sepsis. however, it was comparitively less effective at predicting sepsis, as evidenced by a precision value of 0.14 and a recall value of 0.25, although this was a bit to be expected.

Looking at both the recall value and confusion matrix, our neural network model was able to correctly predict around a quarter of all time steps labeled with Sepsis. Although we recognized that a high recall value is incredibly important in a medical setting, we felt that it is worth pointing out that false positive sepsis diagnoses can be dangerous, as hospital resources can become stretched and we don't want to subject patients to more stress and costs by administering unnecessary treatments. Therefore, maximizing the F1-score based on the prediction throushold provided that key balance between the True Positive and False positive rates.

<img src='/images/sepsis-threshold.png'>

I took advantage of the sigmoid function generating a vector of probabilities to calculate the F1-score at different prediction threshold values. Decreasing the threshold of being classified as Sepsis inevitably gave us higher recall, but it came at a hefty cost of increasing the False Positive rate, as well. Graphing out the relationship between the threshold and F1-score allowed us to figure out where the F1-score was maximized, and we used that threshold value on the test set predictions to receive a final F1-score of 0.173

Conclusions
------

We initially faced a heavily imbalanced dataset, with only around 2% of all patients having sepsis at some point in their stay. Of these, 32% of them were correctly identified wth sepsis with our neural network classification model. For the remaining 98% of patients without sepsis, we also correctly labeled 98% of them! Considering how we tackled a dataset with 65% of it missing, we think that our data imputation methods and SMOTE implementation was quite successful. We do leave a lot of room for improvement, such as better feature engineering, exploring different models, and moving our work to cloud computing.

As the competition date drew closer, my team and I rehearsed our presentation over and over again, making sure that we were fluid in our delivery and unwavering in our voices. We stopped often to give each other feedback and bounced off of each other without feeling like we were ignored. This team dynamic was incredibly refreshing and we went into that morning feeling extra confident.

Although our model was not technically the best-performing across all teams, the judges were incredibly impressed by how my team focused on bringing technical material down to a level where the audience could understand our methods, keeping them engaged and being able to answer every question, and showing our preparedness. We were awarded 1st place after some heart-pounding suspense, marking the end of a wonderful experience.


<figure>
  <img src="/images/sepsis-photos1.jpeg" alt="Photo of everyone">
  <figcaption>Group photo of BC Case Competition 2024. From left to right: Sasha Tomic, Doc. Arvind Sharma, Angelo Marinaro, Trevor Petrin, Pin Lyu, Alan Lin, David Weinberg, Debashis Rana, Doc. Larry Fulton, Matthew Williams</figcaption>
</figure>

<figure>
  <img src="/images/sepsis-photos2.jpg" alt="Photo of my team">
  <figcaption>My team! So proud of everyone.
   From left to right: Trevor Petrin, Alan Lin, Angelo Marinaro, and Pin Lyu. </figcaption>
</figure>

<figure>
  <img src="/images/sepsis-photos3.jpg" alt="Photo of Me">
</figure>

A big thank you to Boston College for setting up this event for us to gain valuable experience in public speaking, working as a team, and applying our machine learning skills to a highly relevant real-world task. To Doc. Larry and Doc. Arvind, thank you for your mentoring and advice throughout this project. And thank you to you, the reader, for sticking with me through this article! I'd love to hear your thoughts.