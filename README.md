# DOG BREED IDENTIFICATION
**Problem Statement** - To predict/ identify the name of the breed based on the image of the dog.

**Overview:**

- Image recognition and classification have successfully applied in various domains, such as face recognition and scene understanding of autonomous driving.
- At present, human face identification is successfully used for authentication and security purposes in many applications.
- Therefore, there are attempts to extend studies from human to animal recognition. In particular, dogs are one of the most common animals.
- Since there are more than 180 dog breeds, dog breed recognition can be an essential task in order to provide proper training and health treatment.
- Previously, dog breed recognition is done by human experts. However, some dog breeds might be challenging to evaluate due to the lack of experts and the difficulty of breeds' patterns themselves. It also takes time for each evaluation.
- The main objective of this case study is to create a classifier using CNN capable of determining a dog’s breed from an image.

**Dataset:**

- In this competition, a strictly canine subset of ImageNet has been provided in order to practice fine-grained image categorization.
- We have been provided with training set and a test set of images of dogs and there are total of 20581 files.
- Each image has a filename that is its unique id.
- The data set contains 120 breeds of dogs.

*Source - [Dog Breed Identification Data](https://www.kaggle.com/competitions/dog-breed-identification/data)*

**Classification Metric:**
- For each image in the test data, we must predict a probability for each of the different breeds as part of the case study.
- Multi class log loss between the predicted probability and the observed target has been used as the metric.

**Results:**

| Model | Train Loss | Cross-Validation Loss | Train Accuracy | Cross-Validation Accuracy |
|----------|----------|----------|----------|----------|
| InceptionV3    | 0.2517   | 0.2554   | 0.9229   | 0.9188   |
| InceptionV3+ResNet152    | 0.2019 | 0.2531 | 0.9397  | 0.9226   |

From the above results, it is evident that the stacked model (InceptionV3 + ResNet152) performed better when compared to InceptionV3 model.

For more information on this case study, please read my blog - [Dog Breed Identification](https://medium.com/@SharathKumarVadla/dog-breed-identification-98cb9f7bd815)
