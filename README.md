# Pathology report classification

A pathology report is written using technical medical language by a pathologist to inform the referring doctor about cancer diagnostic and staging information. Aim of pathology report is to disseminate the results from pathologists' inspection back to the physician or surgeon. It consists of information like patients name, gross description, microscopic description, diagnosis, tumor size and margin including the other information divided into three to four sections like medical history, diagnosis, summary and conclusion. For more information about pathology reports read [here](https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/pathology-reports-fact-sheet#:~:text=The%20pathologist%20sends%20a%20pathology,explain%20the%20report%20to%20them). However, the pathology reports aren't structured and lack one or the other of the abovementioned sections except the diagnostic information. On top of that, these reports are handwritten notes on a printed template, signed and stored as digital pdf documents. Converting the pdf documents to machine readable text files using optical character recognition tools adds additional noise to these otherwise unstructured reports. We propose an approach to utilize these highly-noisy and unstructured pathology reports to automatically extract information about gleason grading.

We presented the work at AIDP2021 (International Workshop on Artificial Intelligence for Digital Pathology) workshop colocated at ICPR2021 (International Conference on Pattern Recognition). 


Official Abstract: Free-text reporting has been the main approach in clinical pathology practice for decades. Pathology reports are an essential information source to guide the treatment of cancer patients and for cancer registries, which process high volumes of free-text reports annually. Information coding and extraction are usually performed manually and it is an expensive and time-consuming process, since reports vary widely between institutions, usually contain noise and do not have a standard structure. This paper presents strategies based on natural language processing (NLP) models to classify noisy free-text pathology reports of high and low-grade prostate cancer from the open-source repository TCGA (The Cancer Genome Atlas). We used paragraph vectors to encode the reports and compared them with n-grams and TF-IDF representations. The best representation based on distributed bag of words of paragraph vectors obtained an   f1 -score of 0.858 and an AUC of 0.854 using a logistic regression classifier. We investigate the classifier’s more relevant words in each case using the LIME interpretability tool, confirming the classifiers’ usefulness to select relevant diagnostic words. Our results show the feasibility of using paragraph embeddings to represent and classify pathology reports.

The publication could be found [here]() and its related slides could be found [here](http://prisca.unina.it/aidp2020/07.pdf) and [here](https://www.slideshare.net/IIG_HES/classification-of-noisy-freetext-prostate-cancer-pathology-reports-using-natural-language-processing-nlp-anjani-k-dhrangadhariya-hesso-valaiswallis-aidp2021-workshop-colocated-at-icpr2021).

The work was kindly funded by EU project [ExaMode](https://www.examode.eu/).

If you use the work please cite it using the following:

```
@inproceedings{DBLP:conf/icpr/DhrangadhariyaO20,
author = {Anjani Dhrangadhariya and
Sebastian Ot{\'{a}}lora and
Manfredo Atzori and
Henning M{\"{u}}ller},
title = {Classification of Noisy Free-Text Prostate Cancer Pathology Reports
Using Natural Language Processing},
booktitle = {Pattern Recognition. {ICPR} International Workshops and Challenges
- Virtual Event, January 10-15, 2021, Proceedings, Part {I}},
series = {Lecture Notes in Computer Science},
volume = {12661},
pages = {154--166},
publisher = {Springer},
year = {2020},
url = {https://doi.org/10.1007/978-3-030-68763-2\_12},
doi = {10.1007/978-3-030-68763-2\_12},
timestamp = {Tue, 23 Mar 2021 14:14:21 +0100},
biburl = {https://dblp.org/rec/conf/icpr/DhrangadhariyaO20.bib},
bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
