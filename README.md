# Skin lesion classification with crowdsourced similarities:
Abstract:

In recent time a great potential for medical image analysis has been discovered and developed within machine learning methods, especially convolutional neural networks (CNNs). However, obtaining annotated datasets is both time consuming and demands a high level of expertise, therefore, this paper examines an alternative approach of annotation called crowdsourcing. Crowdsourcing is a method that relies on “outsourcing of tasks to a crowd of individuals” and it has been used for annotating large datasets of both non-medical and medical images. In this study, we examine how the crowdsourcing platform called Amazon Mechanical Turk can be used to gather similarity assessments of medical images of skin lesions, and create crowd annotated datasets to improve already developed machine-learning models. For the purpose of defining valid and invalid crowdsourced data, the following metrics are defined: similarity rating, confidence-score, annotation number, task completion time and workers' approval rating. The applied data-extraction activities include the removing of impossible triplets based on the similarity assessments, excluding results with a confidence-score of 'very uncertain', 'uncertain' and 'very confident', as well as excluding annotations if the first HIT completed by a worker is done in less than 60 seconds, or if the subsequent HITs completed are done in less than 10 seconds or more than 200 seconds. Leveraging all the metrics, the overall accuracy of the crowdsourced data is increased by approximately  +20\%-points. Further, our findings indicate that a complete retraining of an existing CNN-based machine-learning model including the created crowd labels is performing better measured by AUC compared to incrementally training the model with the created crowd labels. Finally, we discuss both the possibilities and challenging aspects related to our findings that requires further research within the the domain of using crowdsourced similarity ratings of medical images of skin lesions. All data and models are available in this GitHub-repository.

# Our Google-Colab used to plot the data used in our report:
Plotting of the raw batch results:  
https://colab.research.google.com/drive/1GfNeLsTTkTeM1smPC3a6JgJqyIRsRblw
  
  
Filtering and plotting of the data after the data-extraction:  
https://colab.research.google.com/drive/1GhXObN568yiKT7YIjmjbXH-5Rbcxtfrd
  
  
Plotting of the ML-models AUC:  
https://colab.research.google.com/drive/1uN2fwVSb26FGLJd1OBXjdGYyXVtgLoPG
  
  
# Code for the used ML-model is developed by the authors who have the following GitHub-repo:
https://github.com/raumannsr/ENHANCE/

