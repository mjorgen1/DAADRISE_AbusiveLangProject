# Multi-Class Detection of Abusive Language on Twitter Using Automated Machine Learning

Mackenzie Jorgensen, Minho Choi, Marco Niemann, Jens Brunk, & Jorg Becker

<p align="center">
</p>

Abusive language detection is a difficult task for comment moderators. Machine Learning has shown promising results in detecting abusive language online. We aim to explore an underdeveloped field of Automated Machine Learning (Auto-ML) for text classification with the application of abusive language detection on Twitter. We utilize an English data-set (from Davidson et. al) and a German data-set (from Wiegand et al.). We propose Automated Machine Learning with multi-class classification as an approach to abusive language detection online. We have a pending publication of our work in the 15th International Business Informatics Congress.

We owe a great deal to Davidson et al. (data and code) and Wiegand et al.'s (data) work. We extended Davidson's code here by adding H2O-Auto-ML, German language functionality for pre-processing and feature extraction, and feature selection and undersampling methods. 

## Links
\- [Research paper final version published](https://library.gito.de/oa_wi2020-r7.html)
\- [Davidson et al.'s GitHub Repository](https://github.com/t-davidson/hate-speech-and-offensive-language)
\- [Wiegand et al.'s GitHub Repository](https://github.com/uds-lsv/GermEval-2018-Data)

## Execution
To execute the German run, go to /H2o-Work/Germ_H2OAutoML-ULTRA.ipynb and you can see comments relating to feature selection algorithms and undersampling techniques. You can choose which methods you would like to run by commenting out the ones you do not want and uncommenting the ones you would like to use. Further, you will have to save the German data-set from Wiegand et al.'s GitHub and update your folders in the code with regard to the data and saving models and results. Then you can run the Jupyter Notebook on your machine. 

To execute the English run, go to /H2o-Work/EnglishH2OAutoML-ULTRA.ipynb and you can see comments relating to feature selection algorithms and undersampling techniques. You can choose which methods you would like to run by commenting out the ones you do not want and uncommenting the ones you would like to use and importing different undersampling methods. Further, you will have to save the English data-set from Davidson et al.'s GitHub and update your folders in the code with regard to the data and saving models and results. Then you can run the Jupyter Notebook on your machine. 

