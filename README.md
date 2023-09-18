<h1 align="center"><font size="5">Predictive Modeling of Near-Earth Objects using Artificial Neural Networks</font></h1>

<p align="center">
  <img src="https://www.techexplorist.com/wp-content/uploads/2021/12/NASAs-Eyes-on-Asteroids.jpg" alt="Techexplorist">
</p>

<p>
  Picture Source: <a href="https://www.techexplorist.com/wp-content/uploads/2021/12/NASAs-Eyes-on-Asteroids.jpg">Techexplorist</a>
</p>

<br>

Welcome to the "Predictive Modeling of Near-Earth Objects using Artificial Neural Networks" repository! In this project, we ventured into the fascinating realm of celestial objects and harnessed the power of artificial neural networks to tackle a critical challenge. Our mission? To predict the classification of Near-Earth Objects (NEOs) as either "hazardous" or "non-hazardous." NEOs are asteroids and comets that come close to Earth's orbit, and understanding their potential threat is paramount. With the aid of advanced machine learning techniques, we embarked on a journey to build a predictive model that could identify hazardous NEOs. Armed with data and artificial intelligence, we delved into the cosmos to enhance our ability to safeguard our planet.

<br>

## Near-Earth Objects (NEOs)

Near-Earth objects (NEOs) are asteroids or comets of various sizes that come close to Earth's orbit. Of the more than 600,000 known asteroids in our Solar System, more than 20,000 are NEOs. These objects can potentially pose a risk to our planet, depending on their size. Understanding NEOs is crucial for planetary defense and space science.

## About NEOs

NEOs could potentially impact our planet, causing varying degrees of damage. While the chance of a large object hitting Earth is small, it could have catastrophic consequences. NEOs are actively monitored and tracked to:

- Become aware of the current and future positions of NEOs relative to Earth.
- Estimate the likelihood of Earth impacts.
- Assess the consequences of potential impacts.
- Develop NEO deflection methods.

The NEO Segment observes NEOs, predicts their orbits, produces impact warnings when necessary, and participates in potential mitigation measures.

## Keywords

- Classification
- Science and Technology
- Astronomy
- Binary Classification
- Near-Earth Objects
- Artificial Neural Networks

<br>

## Context

Despite the vast distances in outer space, some objects are closer than we think. A distance of 70,000 kilometers may seem insignificant, but in astronomical terms, it's relatively close and can disrupt natural phenomena. These objects, such as asteroids, have the potential to harm our planet. Hence, it is wise to monitor and understand the NEOs in our vicinity.

<br>

## Statement

In this project, we used artificial neural networks (ANNs) to perform binary classification to predict whether Near-Earth Objects (NEOs) are hazardous or not. All the details and code can be found in the [NASA_NEOs_ANN_Classification.ipynb](https://github.com/doguilmak/Nearest-Earth-Objects-Classification/blob/main/NASA_NEOs_ANN_Classification.ipynb) file.

![Accuracy and Loss](acc_loss.png)

<br>

## Dataset

The dataset contains 10 different columns. We created a binary classification model to predict the "hazardous" output using features such as estimated diameter, relative velocity, miss distance, and absolute magnitude. For more information about the dataset, you can visit:

- [Dataset on Kaggle](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects?select=neo.csv)
- [NASA API](https://api.nasa.gov/)
- [NEO Earth Close Approaches](https://cneos.jpl.nasa.gov/ca/)

<br>

## How to Run Notebook

You can run the project by following these steps:

1. Clone this repository to your local machine.
2. Open the [NASA_Near_Earth_Objects-ANN_Classification.ipynb](https://github.com/doguilmak/Nearest-Earth-Objects-Classification/blob/main/NASA_Near_Earth_Objects-ANN_Classification.ipynb) Jupyter Notebook file.
3. Run the code cells in the notebook to execute the classification model and view the results.

<br>

## Conclusion

The binary classification model trained to predict whether Near-Earth Objects (NEOs) are hazardous or not has yielded promising results. The evaluation metrics reveal the following key insights:

-   **Accuracy**: The model achieved an accuracy score of approximately 87.92%, indicating that it correctly classified a significant portion of NEOs.
    
-   **Precision**: With a precision score of approximately 81.16%, the model demonstrates its ability to identify hazardous NEOs with a relatively low rate of false positives. This is essential for ensuring that resources are appropriately allocated to address potential threats.
    
-   **Recall**: The recall score, measuring around 98.76%, signifies the model's effectiveness in capturing a vast majority of actual hazardous NEOs. It minimizes the risk of failing to detect dangerous objects.
    
-   **Specificity**: The specificity score of approximately 77.09% reflects the model's capability to correctly identify non-hazardous NEOs. This is crucial for preventing unnecessary alarm or resource allocation for harmless objects.
    
-   **F1 Score**: The F1 score, which combines precision and recall, stands at approximately 89.10%. This balanced metric highlights the model's ability to strike a compromise between minimizing false alarms and ensuring high detection rates.

In summary, the classification model, leveraging Artificial Neural Networks (ANNs), demonstrates a solid performance in identifying hazardous Near-Earth Objects. Its accuracy, precision, and recall indicate its potential utility in early detection and tracking efforts, contributing to the ongoing mission of safeguarding our planet from potential impacts.

## Contact Me

If you have any questions, feedback, or suggestions, please feel free to reach out:

- Twitter: [@Doguilmak](https://twitter.com/Doguilmak)
- Email: doguilmak@gmail.com
