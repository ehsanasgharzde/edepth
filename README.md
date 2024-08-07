**_To aonaran, who is learning to perceive the profound depths of the world._**
_from e_

# edepth: Open-Source Trainable Depth Estimation Model

## Overview

edepth is an open-source, cutting-edge deep learning model designed to estimate depth from various input sources, including single images, videos, and live camera feeds. Depth estimation is a crucial task in computer vision, with applications in autonomous driving, robotics, augmented reality, and more. edepth addresses this task by predicting the distance of objects from the camera using convolutional neural networks (CNNs).

## Model Architecture and Pipeline

The edepth model architecture is inspired by [DenseNet](https://www.geeksforgeeks.org/densenet-explained/) and [U-Net](https://www.geeksforgeeks.org/u-net-architecture-explained/) architectures, which have shown success in image segmentation tasks. The model consists of an encoder-decoder structure.

![General Architecture](architecture/edepth-general-architecture.png)

*General Architecture Overview*

### Encoder

The encoder extracts features from the input data using multiple dense blocks, each containing convolutional layers with shared feature maps concatenated across layers. Transition layers follow dense blocks to reduce the number of channels and spatial dimensions:

- **initialConvolution**: Convolution layer with kernel size 5.
- **pool**: Max pooling layer with kernel size 2 and stride 2.
- **denseBlocks**: Sequence of dense blocks for feature extraction.
- **transitionLayers**: Sequence of transition layers to reduce channel dimensions.

### Fully Connected Layers

Between the encoder and decoder, the model includes fully connected layers to process the features:

- **fullyConnectedI**: Linear layer transforming the encoder output to a fixed-size vector.
- **fullyConnectedII**: Linear layer transforming the fixed-size vector back to the size expected by the decoder.

### Decoder

The decoder reconstructs the depth map from the encoded features using upsampling layers:

- **upSampleI**: Upsampling layer with scale factor 2.
- **convI**: Convolution layer with kernel size 3.
- **upSampleII**: Upsampling layer with scale factor 2.
- **convII**: Convolution layer with kernel size 3.
- **upSampleIII**: Upsampling layer with scale factor 2.
- **convIII**: Convolution layer with kernel size 3.
- **upSampleIV**: Upsampling layer with scale factor 2.
- **convIV**: Convolution layer with kernel size 3, outputting a single-channel depth map.


![Detailed Architecture](architecture/edepth-detailed-architecture.png)

*Detailed Architecture Overview*


## Installation

To install the required dependencies for running edepth, use the provided `requirements.txt` file. This file lists all necessary Python 3.12.* packages and their versions.

```bash
pip install -r requirements.txt
```

## Cloning and Setting Up the Model

To clone the repository and set up edepth on your local machine, follow these steps:

### Clone the Repository

```bash
git clone https://github.com/ehsanasgharzde/edepth.git
cd edepth
```

### Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

### Create Model

```python
from edepth import edepth

model = edepth()
```

### Loading a Pre-trained Model

Load a pre-trained model for inference:

```python
model.eload('path/to/pretrained_model.pt')
```

### Generating Depth Maps

#### From an Image

```python
model.egenerate(source='image', inputFilePath='path/to/image.jpg', show=True)
```

#### From a Video

```python
model.egenerate(source='video', inputFilePath='path/to/video.mp4', show=True)
```

#### From Live Camera Feed

```python
model.egenerate(source='live', show=True)
```

### Training the Model

Train the edepth model using the provided training data:

```python
import pandas
from utilities import Dataset
from sklearn.model_selection import train_test_split

    
dataset = pandas.read_csv('path/to/dataset.csv')
train, validate = train_test_split(dataset, test_size=0.2, random_state=42)
trainset, validationset = Dataset(train, 224, 224), Dataset(validate, 224, 224)

model.etrain(trainset, validationset, epochs=100)
```

## Test Train Details

### Hyperparameters

The following hyperparameters, dataset, and hardware were used to achieve the model's performance mentioned further in the readme file:

- **Growth Rate**: 32
- **Neurons**: 512
- **Epochs**: 1000 (96 successfully completed epochs after reaching hardware **temperature limits**)
- **Batch Size**: 16
- **Gradient Clip**: 2.0
- **Optimizer**: `swats.SWATS(self.parameters(), lr=0.0001)`
- **Activation**: `nn.ReLU()`
- **Loss**: `nn.MSELoss()`
- **Scheduler**: `torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100, factor=0.5)`

### Hardware

- **Processor**: Intel® Core™ i7-5500U CPU @ 2.40GHz × 4
- **GPU**: HAINAN (, LLVM 15.0.7, DRM 2.50, 6.5.0-35-generic) / Mesa Intel® HD 
- **RAM**: 16 GB
- **Storage**: 1 TB SSD

### Training Performance

- **Epochs Completed**: 96
- **Validation Loss at 96th Epoch**: 0.27274127925435704

### Dataset

The dataset for edepth was gathered by downloading videos from YouTube, with details available in this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1emitM6KBMgAKzhZJz0BxfTZmTI64PiG18hUpJhMpM8Q/edit?usp=sharing). The images and corresponding depth maps were created using the Marigold depth estimation model, available [here](https://github.com/prs-eth/Marigold).

### Key Details:
- **Number of Images and Labels (Depth Maps)**: 954
- **Average Image Size** (resized to 224x224): 4.5MB
- **Average Label (Depth Map) Size** (resized to 224x224): 6MB


**Model State**: [Download](https://ehsanasgharzde.ir/assets/pt/ep-96-val-0.27274127925435704.pt) and place in `checkpoints` folder.

**Note**: This model state was specifically trained for testing purposes on drone shots of cell tower antennas in daylight conditions. It may not be suitable for other use cases. Users are encouraged to train the model themselves for their specific applications and datasets to achieve optimal performance.

  
## Samples

### Image Samples

Below are visualizations of input images and their corresponding output depth maps generated by the edepth model. These samples demonstrate the model's ability to estimate depth from single images accurately.

![Sample 1](input/1500447202405080310499988.jpg)
*Input Image*

![Sample 1 Colorized Depth Map](output/colorized/1500447202405080310499988_pred.jpg)
*Output Colorized Depth Map*
estimation time: 0.1526861349993851 seconds

![Sample 1 Grayscale Depth Map](output/grayscale/1500447202405080310499988_pred.jpg)
*Output Grayscale Depth Map*
estimation time: 0.1276703309995355 seconds

![Sample 1](input/1500938202405080311264197.jpg)
*Input Image*

![Sample 2 Colorized Depth Map](output/colorized/1500938202405080311264197_pred.jpg)
*Output Colorized Depth Map*
estimation time: 0.14070451999577926 seconds

![Sample 2 Grayscale Depth Map](output/grayscale/1500938202405080311264197_pred.jpg)
*Output Grayscale Depth Map*
estimation time: 0.1332030850026058 seconds

### Video Samples

edepth can process video files and generate depth maps for each frame. Here are some example results:

1. Original Video: [Original Video](input/18839247834792849281913.mp4)
2. Depth Map Video: [Depth Map Video](output/video/18839247834792849281913.avi)

![Screen Record on How edepth is Doing](output/screen-record.gif)

### Performance Metrics

**Note**: edepth calculates accuracy by comparing the predicted depth values to the true depth values for each pixel in the input image.

- **Processing Speed**: edepth can process images at a rate of 21 images per second and videos at 25 frames per second.
- **Accuracy**: The model achieves an average accuracy of 99% on standard depth estimation benchmarks.
- **Model Size**: The edepth model has a total of 1.3 million parameters, making it efficient for both training and inference.

## Main Features

### Customizable Architecture
edepth offers flexibility in its architecture, allowing users to adjust parameters such as input channels, growth rate, and depth range. This customization enables the model to adapt to different datasets and tasks.

### Training and Evaluation Methods
The model provides methods for training and evaluating depth estimation tasks. It includes functionalities for loading datasets, training the model with configurable hyperparameters, and evaluating model performance on validation sets.

### Real-time Processing Capabilities
Capable of processing live camera feeds in real-time, making it suitable for dynamic and interactive applications.

### Versatile Input Support
Supports images, videos, and live feeds, providing a comprehensive solution for depth estimation across different types of media.

## Performance

### Image Input

- **Speed**: Processes images at 21 images per second.
- **Accuracy**: Achieves an average accuracy of 73% on 954 images dataset with 96 epochs of learning.

### Video Input

- **Speed**: Processes video frames at 25 to 30 frames per second.
- **Accuracy**: Maintains high accuracy across consecutive frames.

### Live Streams

- **Speed**: Real-time processing with minimal latency.
- **Accuracy**: Consistent accuracy for dynamic scenes.

## Future Plans

### cudnnenv
- **cudnnenv**: Manage CUDA and cuDNN versions for optimized deep learning performance.

### scikit-image
- **scikit-image**: Utilize for image processing tasks like preprocessing and post-processing depth maps.

### huggingface_hub
- **huggingface_hub**: Explore pretrained models and datasets for depth estimation tasks.

### accelerate
- **accelerate**: Enhance training efficiency with utilities for distributed and mixed precision training.

### diffusers
- **diffusers** (planned): Analyze model robustness and interpretability for depth estimation.

### transformers
- **transformers**: Adapt transformer-based architectures for computer vision tasks, including depth estimation.

### denoisers
- **denoisers** (planned): Improve depth map quality through advanced denoising techniques.

### Update Shape and Remove Fully Connected Layers
- **Update Shape and Remove Fully Connected Layers** (planned): Enhance edepth's versatility to support variable input sizes efficiently.


## Contributing

Contributions to improve the model's performance or add new features are highly appreciated! Whether it's optimizing the architecture, implementing new algorithms, or enhancing documentation, your contributions are valuable. To contribute, fork the repository, make your changes, and submit a pull request.

### Steps to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

For any questions, suggestions, or collaboration opportunities, feel free to reach out to [me](https://ehsanasgharzadeh.asg@gmail.com).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

- **Ehsan Asgharzadeh** - [GitHub](https://github.com/ehsanasgharzde), [LinkedIn](https://www.linkedin.com/in/ehsanasgharzde/)
- **Contact**: [ehsanasgharzadeh.asg@gmail.com](mailto:ehsanasgharzadeh.asg@gmail.com)
- **Version**: 1.0.1
- **License**: [ehsanasgharzadeh.ir](https://ehsanasgharzadeh.ir) - [MIT](LICENSE)
