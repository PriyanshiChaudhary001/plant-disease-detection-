# Plant Disease Detection (Basic Machine Learning)

An educational plant-leaf image classification project built with
Python, NumPy, and OpenCV. This updated version uses HSV colour
histograms and a nearest-class-centroid approach to classify leaf
images. It does not use a neural network or scikit-learn.

The classifier compares an image's colour-histogram features with the
class centroids learned during training and returns the closest class.
Results are experimental and should not be treated as a reliable
plant-disease diagnosis.

## Features

-   Extracts HSV colour-histogram features from leaf images using
    OpenCV.
-   Learns a centroid for each class from the training images.
-   Predicts the closest class for a separate leaf image.
-   Supports common image formats such as JPG and PNG.

## Project Structure

``` text
plant-disease-ml/
├── data/
│   ├── Tomato___Early_blight/
│   │   ├── leaf1.jpg
│   │   └── leaf2.png
│   ├── Tomato___Late_blight/
│   │   ├── leaf1.jpg
│   │   └── leaf2.png
│   └── Tomato___healthy/
│       ├── leaf1.jpg
│       └── leaf2.png
├── train_basic.py
├── predict_basic.py
├── requirements.txt
└── README.md
```

The `data/` folder and its image dataset are not included in this
repository. Add your own correctly labelled dataset. Each class folder
should contain images belonging to that class. Folder names are used as
class labels.

## Requirements

-   Python 3
-   NumPy
-   OpenCV (`opencv-python`)

Install the dependencies from the project directory:

``` bash
pip install -r requirements.txt
```

If you are setting up only the basic classifier and your requirements
file does not list these packages, install them with:

``` bash
pip install numpy opencv-python
```

## Dataset

You can use a labelled plant-leaf dataset such as [PlantVillage on
Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
The dataset is not bundled with this repository.

Organize the images so the selected dataset directory directly contains
one subfolder per class. For example:

``` text
data/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
└── Tomato___healthy/
```

Use the correct labels from the dataset. Do not place unrelated images
in a class folder. If your downloaded dataset has additional `train`,
`valid`, or nested directories, set `--data_dir` to the directory that
directly contains the class folders.

## Training

Run the following command from the `plant-disease-ml` directory:

``` bash
python train_basic.py --data_dir data --filter tomato --max_per_class 100
```

-   `--data_dir` specifies the directory containing the class folders.
-   `--filter tomato` selects tomato-related classes.
-   `--max_per_class 100` limits the number of images used per class.

Adjust the data directory and options to match your dataset. The script
will report its training outcome and save the model artifact according
to its implementation. Keep the generated model file so the prediction
script can load it.

## Prediction

Place a separate test image in the project directory, outside the
training `data/` folders. Run:

``` bash
python predict_basic.py --image sample_leaf.png
```

Replace `sample_leaf.png` with the exact name or path of your image.
Both `.png` and `.jpg` files are supported. The prediction script uses
the model generated during training.

## How It Works

1.  Open and process each leaf image with OpenCV.
2.  Extract an HSV colour histogram and use it as a numeric feature
    representation.
3.  Calculate a centroid for each labelled class from its training
    features.
4.  Compare a test image's features with the learned centroids and
    predict the nearest class.

This is a simple baseline classifier. It focuses on colour information
and does not perform the HOG texture analysis or Random Forest
classification used by the older version.

## Results

Record your actual evaluation results here after training and testing.
Do not assume that a training metric represents accuracy on new images.

``` text
Evaluation accuracy: Add your measured result here
```

## Limitations

-   This is an educational demonstration, not a dependable agricultural
    diagnostic system.
-   Colour-histogram features alone may not distinguish diseases with
    similar colours or symptoms.
-   Results can vary with lighting, backgrounds, camera quality, leaf
    variety, and dataset quality.
-   Dataset performance may not generalize to real field photographs.
-   Any treatment or crop-management decision should be confirmed with a
    qualified local agriculture expert.

## License

Add a license before distributing or reusing this project publicly.

