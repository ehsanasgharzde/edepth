import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from typing import Any, Optional, Union
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler  as Scheduler
import torch.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from alive_progress import alive_bar

import time
import random


class edepth(nn.Module):
    """
    Depth Estimation Convolutional Neural Network

    edepth is a convolutional neural network designed for estimating depth maps from input images.
    It employs an encoder-decoder architecture, where the encoder extracts features from the input
    image, and the decoder reconstructs the depth map from these features.

    Inputs:
        - Input images: RGB images with shape (batchSize, channels, height, width).

    Outputs:
        - Depth maps: Estimated depth maps with shape (batchSize, height, width).

    Usage:
        To use the edepth model, create an instance of the class and call it with input images.
        You can also train the model, evaluate its performance, generate depth maps from images
        or videos, and visualize the results.

    Author: Ehsan Asgharzadeh
    Copyright: Copyright (C) 2024 Ehsan Asgharzadeh
    License: https://ehsanasgharzde.ir
    Contact Info: https://ehsanasgharzadeh.asg@gmail.com
    Version: 0.0.1
    """

    __author__ = "Ehsan Asgharzadeh"
    __copyright__ = "Copyright (C) 2024 Ehsan Asgharzadeh"
    __license__ = "https://ehsanasgharzde.ir"
    __contact__ = "https://ehsanasgharzadeh.asg@gmail.com"
    __version__ = "0.0.1"

    def __init__(self, 
                 optimizer: Any = None,
                 activation: Any = None,
                 loss: Any = None,
                 scheduler: Any = None,
                 dropout: Any = None,
                 numLayers: int = 5, 
                 kernelSize: int = 3, 
                 numKernels: tuple | int = [16, 32, 64, 128, 256], 
                 inputChannels: int = 3,
                 outputChannels: int = 1,
                 strideLength: int = 1, 
                 poolSize: int = 2, 
                 numNeurons: int = 1024, 
                 minDepth: float = None,
                 maxDepth: float = None,
                 linBiasI: bool = True,
                 linBiasII: bool = True,
                 padding: int = 1) -> None:
        """
        Initializes the depth estimation model.

        Args:
            `numLayers` (int, optional): Number of layers in the encoder and decoder. Defaults to 5.
            `kernelSize` (int, optional): Size of the convolutional kernels. Defaults to 3.
            `numKernels` (int, optional): Number of kernels in the convolutional layers. Defaults to 64.
            `strideLength` (int, optional): Stride length for convolutional layers. Defaults to 1.
            `poolSize` (int, optional): Size of the max-pooling windows. Defaults to 2.
            `numNeurons` (int, optional): Number of neurons in the fully connected layers. Defaults to 512.
            `activation` (str, optional): Activation function to be used. Defaults to 'relu'.
            `minDepth` (float, optional): Minimum depth value for depth estimation. Defaults to None.
            `maxDepth` (float, optional): Maximum depth value for depth estimation. Defaults to None.
            `linBiasI` (bool, optional): Whether to include bias in the first linear layer. Defaults to True.
            `linBiasII` (bool, optional): Whether to include bias in the second linear layer. Defaults to True.
            `padding` (int, optional): Padding size for convolutional layers. Defaults to 1.
            `inputChannels` (int, optional): Number of input channels for the convolutional layers. Defaults to 64.
            `outputChannels` (int, optional): Number of output channels for the convolutional layers. Defaults to 128.
        """
        super(edepth, self).__init__()

        self.numLayers = numLayers
        self.kernelSize = kernelSize
        self.numKernels = numKernels
        self.strideLength = strideLength
        self.poolSize = poolSize
        self.numNeurons = numNeurons

        self.encoder = self.__buildEncoder(numLayers, kernelSize, numKernels, inputChannels)
        self.decoder = self.__buildDecoder(numLayers, kernelSize, numKernels[-1], outputChannels)
        self.adaptivePool = nn.AdaptiveAvgPool2d((7, 7))


        self.convI = nn.Conv2d(in_channels=inputChannels, out_channels=outputChannels, kernel_size=kernelSize, padding=padding)
        self.convII = nn.Conv2d(in_channels=outputChannels, out_channels=outputChannels * 2, kernel_size=kernelSize, padding=padding)
        self.convIII = nn.Conv2d(in_channels=outputChannels * 2, out_channels=outputChannels * 4, kernel_size=kernelSize, padding=padding)
        
        self.fullyConnectedLayersI = nn.Linear(7 * 7 * outputChannels * 4, numNeurons, bias=linBiasI)
        self.fullyConnectedLayersII = nn.Linear(numNeurons, 1, bias=linBiasII)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.scheduler = scheduler
        self.dropout = dropout

        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        if activation is None:
            self.activation = nn.ReLU()
        if loss is None:
            self.loss = nn.MSELoss()
        if scheduler is None:
            self.scheduler = Scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.minDepth = minDepth
        self.maxDepth = maxDepth
 
    def __del__(self) \
                      -> None:
        """
        Deconstructor to release GPU memory.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __str__(self) \
                      -> str:
        """
        Return a string representation of the object including memory address.
        """
        return f"<class 'edepth' at {hex(id(self))}>"

    def __repr__(self) \
                       -> str:
        """
        Return a detailed string representation of the object.
        """
        return f"Author: Ehsan Asgharzadeh <https://ehsanasgharzde@gmail.com>\nLicense: https://ehsanasgharzde.ir"
   
    def __class__(self) \
                        -> str:
        """Return a custom representation of the class."""
        return "<class 'edepth'>"
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the depth estimation model.

        Args:
            x (torch.Tensor): Input tensor of shape (batchSize, channels, height, width).

        Returns:
            torch.Tensor: Output tensor representing the estimated depth map.

        Raises:
            ValueError: If the input tensor does not have the expected shape.

        Note:
            The input tensor should have the expected shape (batchSize, channels, height, width).
            - batchSize: Number of samples in the batch.
            - channels: Number of input channels (e.g., 3 for RGB images).
            - height: Height of the input images.
            - width: Width of the input images.
        """
        if x.dim() != 4:
            raise ValueError("ValueError | Input tensor must have shape (batchSize, channels, height, width).")

        features = self.encoder(x)
        pooledFeatures = self.adaptivePool(features)
        
        convIoutput = self.convI(pooledFeatures)
        convIIoutput = self.convII(convIoutput)
        convIIIoutput = self.convIII(convIIoutput)
        
        flattenedFeatures = convIIIoutput.view(convIIIoutput.size(0), -1)
        
        fullyConnectedIoutput = self.activation(self.fullyConnectedLayersI(flattenedFeatures))
        if self.dropout is not None:
            fullyConnectedIoutput = self.dropout(fullyConnectedIoutput)
        fullyConnectedIIoutput = self.fullyConnectedLayersII(fullyConnectedIoutput)

        batchSize = x.size(0)
        outputHeight = x.size(2)
        outputWidth = x.size(3)
        outputChannels = 1 

        output = fullyConnectedIIoutput.view(batchSize, outputChannels, outputHeight, outputWidth)

        return output


    def esave(self, 
              filepath: str, 
              suffix: str = 'pt') -> None:
        """
        Save the model to a file.

        Args:
            `filepath` (str): Path to the file where the model will be saved.
            `suffix` (str, optional): Suffix indicating the file format. Defaults to 'pt'.

        Raises:
            `ValueError`: If the specified file suffix is not supported.

        Note:
            Supported file suffixes:
            - `pth`: Save only the model's state dictionary.
            - `pt`: Save the entire model including architecture and parameters.

        """
        suffixes = {'pth': torch.save(self.state_dict(), filepath),
                    'pt': torch.save(self, filepath)}
        if suffix in suffixes:
            suffixes[suffix]
            print("Model saved successfully.")
        else:
            raise ValueError("ValueError | Unsupported file suffix. Choose 'pth' or 'pt'.")
    
    def eload(self,
            filepath: str) -> None:
        """
        Load model parameters from a file.

        Args:
            `filepath` (str): Path to the file containing the model parameters.

        Raises:
            `FileNotFoundError`: If the specified file does not exist.
            `RuntimeError`: If attempting to load parameters from an incompatible model.

        Note:
            Ensure that the architecture of the loaded model matches the current model.
            Use `torch.load(filepath, map_location=torch.device('cpu'))` for loading models saved on GPU to CPU.

        """
        try:
            self.load_state_dict(torch.load(filepath, map_location=self.device))
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"FileNotFoundError | File '{filepath}' not found.")
        except RuntimeError as e:
            raise RuntimeError(f"RuntimeError | Error while loading model: {str(e)}")

    def etrain(self, 
            trainset: Dataset,
            validationset: Optional[Dataset] = None, 
            epochs: int = 10, 
            batchSize: int = 32, 
            shuffle: bool = True, 
            validationMetrics: Optional[Dict[str, Any]] = None, 
            gradientClip: Optional[float] = None, 
            logInterval: Optional[int] = None, 
            earlyStoppingPatience: int = 5, 
            checkpointPath: Optional[str] = None, 
            saveBestOnly: bool = True) -> None:
        """
        Train the depth estimation model.

        Args:
            `trainset` (torch.utils.data.Dataset): The training dataset.
            `validationset` (torch.utils.data.Dataset, optional): The validation dataset. Default is None.
            `epochs` (int, optional): The number of epochs for training. Default is 10.
            `batchSize` (int, optional): The batch size for training. Default is 32.
            `shuffle` (bool, optional): Whether to shuffle the training data. Default is True.
            `validationMetrics` (dict, optional): Metrics to evaluate the validation performance. Default is None.
            `gradientClip` (float, optional): Gradient clipping value. Default is None.
            `logInterval` (int, optional): Interval for logging training progress. Default is None.
            `earlyStoppingPatience` (int, optional): Patience for early stopping. Default is 5.
            `checkpointPath` (str, optional): Path to save model checkpoints. Default is None.
            `saveBestOnly` (bool, optional): Whether to save only the best model checkpoint. Default is True.

        Raises:
            `RuntimeError`: If an error occurs during training.

        Returns:
            None
        """
        try:
            trainLoader = DataLoader(trainset, batch_size=batchSize, shuffle=shuffle)
            validationLoader = DataLoader(validationset, batch_size=batchSize, shuffle=False) if validationset is not None else None
            bestValidationLoss = float('inf')
            epochsWithNoImprove = 0
            
            for epoch in range(1, epochs + 1):
                startTime = time.monotonic()
                trainLoss, trainAccuracy = self.__trainPerEpoch(trainLoader, gradientClip)
                validationLoss, validationAccuracy = self.__evaluatePerEpoch(validationLoader) if validationLoader is not None else (None, None)

                if validationLoss is not None and validationLoss < bestValidationLoss:
                    bestValidationLoss = validationLoss
                    if checkpointPath:
                        torch.save(self.state_dict(), checkpointPath)

                endTime = time.monotonic()
                epochMinutes, epochSeconds = divmod(endTime - startTime, 60)

                print(f'Epoch: {epoch:02} | Epoch Time: {epochMinutes:.0f}m {epochSeconds:.0f}s')
                print(f'\tTrain Loss: {trainLoss:.3f} | Train Accuracy: {trainAccuracy * 100:.2f}%')
                if validationLoss is not None:
                    print(f'\tValidation Loss: {validationLoss:.3f} | Validation Accuracy: {validationAccuracy * 100:.2f}%')

                if validationMetrics is not None:
                    for metricName, metricValue in validationMetrics.items():
                        print(f'\t{metricName}: {metricValue:.3f}')

                if saveBestOnly and validationLoss is not None and validationLoss >= bestValidationLoss:
                    epochsWithNoImprove += 1
                    if epochsWithNoImprove >= earlyStoppingPatience:
                        print("Early stopping triggered.")
                        break
                else:
                    epochsWithNoImprove = 0

                if self.scheduler:
                    if isinstance(self.scheduler, _LRScheduler):
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(validationLoss)
                        else:
                            self.scheduler.step()

                if logInterval is not None and epoch % logInterval == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {trainLoss:.4f}")

        except Exception as e:
            raise RuntimeError(f"RuntimeError | Error during training:\n{str(e)}")

    def etest(self, 
            testset: Dataset,  
            batchSize: int = 32, 
            visualize: bool = False, 
            shuffle: bool = False) -> Tuple[float, float, List[int], List[int]]:
        """
        Evaluate the depth estimation model on a test dataset.

        Args:
            `testset` (torch.utils.data.Dataset): The test dataset.
            `batchSize` (int, optional): The batch size for evaluation. Default is 32.
            `visualize` (bool, optional): Whether to visualize predictions. Default is False.
            `shuffle` (bool, optional): Whether to shuffle the test data. Default is False.

        Returns:
            `Tuple[float, float, List[int], List[int]]`: A tuple containing the average loss, 
                accuracy, list of all predicted labels, and list of all true labels.
        """
        loader = DataLoader(testset, batch_size=batchSize, shuffle=shuffle)
        self.eval()
        totalLoss = 0.0
        totalCorrect = 0
        allPredictions = []
        allLabels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self(inputs)
                loss = self.loss(outputs, labels)
                totalLoss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                totalCorrect += (predicted == labels).sum().item()
                allPredictions.extend(predicted.cpu().numpy())
                allLabels.extend(labels.cpu().numpy())

                if visualize:
                    sampleIndex = np.random.randint(0, len(inputs))
                    inputImage = inputs[sampleIndex].cpu().numpy().transpose((1, 2, 0))
                    trueLabel = labels[sampleIndex].item()
                    predictedLabel = predicted[sampleIndex].item()

                    plt.imshow(inputImage)
                    plt.title(f'True Label: {trueLabel} | Predicted Label: {predictedLabel}')
                    plt.show()

        averageLoss = totalLoss / len(testset)
        accuracy = totalCorrect / len(testset) * 100
        print(f'Test Loss: {averageLoss:.4f} | Test Accuracy: {accuracy:.2f}%')

        return averageLoss, accuracy, allPredictions, allLabels

    def egenerate(self,
                source: str,
                inputWidth: int = 640,
                inputHeight: int = 480,
                show: bool = False,
                save: bool = True,
                outputDir: str = 'depthmaps',
                outputFormat: str = 'jpg',
                outputFilename: str = 'depthmap',
                colormap: str = 'colorized',
                frameRange: tuple = None,
                resize: tuple = None) -> None:
        """
        Generate depth maps from images, videos, or live streams.

        Args:
            `source` (str): Source type, can be 'image', 'video', or 'live'.
            `inputWidth` (int, optional): Input image width. Default is 640.
            `inputHeight` (int, optional): Input image height. Default is 480.
            `show` (bool, optional): Whether to display generated depth maps. Default is False.
            `save` (bool, optional): Whether to save generated depth maps. Default is True.
            `outputDir` (str, optional): Directory to save the depth maps. Default is 'depthmaps'.
            `outputFormat` (str, optional): Output image format. Default is 'jpg'.
            `outputFilename` (str, optional): Filename for the saved depth maps. Default is 'depthmap'.
            `colormap` (str, optional): Colormap to use for visualization. Default is 'colorized'.
            `frameRange` (tuple, optional): Range of frames to process in a video. Default is None.
            `resize` (tuple, optional): Resize dimensions for the input frames. Default is None.

        Returns:
            None
        """
        if save and not os.path.exists(outputDir):
            os.makedirs(outputDir)

        if source == 'image':
            image = cv2.imread(source)
            if resize:
                image = cv2.resize(image, (resize[1], resize[0]))
            depthMap = self.__processImage(image, inputWidth, inputHeight, colormap)

            if show:
                cv2.imshow('Depth Map', depthMap)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save:
                outputPath = os.path.join(outputDir, f'{outputFilename}.{outputFormat}')
                cv2.imwrite(outputPath, depthMap)
        
        elif source == 'video':
            cap = cv2.VideoCapture(source)
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = None
            if save:
                outputPath = os.path.join(outputDir, f'{outputFilename}.avi')
                out = cv2.VideoWriter(outputPath, 
                                    cv2.VideoWriter_fourcc(*'DIVX'), 
                                    fps, 
                                    (frameWidth, frameHeight))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frameRange and cap.get(cv2.CAP_PROP_POS_FRAMES) < frameRange[0]:
                    continue
                if frameRange and cap.get(cv2.CAP_PROP_POS_FRAMES) > frameRange[1]:
                    break
                
                if resize:
                    frame = cv2.resize(frame, (resize[1], resize[0]))
                
                depthMap = self.__processImage(frame, inputWidth, inputHeight, colormap)

                if show:
                    cv2.imshow('Depth Map', depthMap)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save:
                    out.write(depthMap)

            cap.release()
            if save:
                out.release()
            cv2.destroyAllWindows()
        
        elif source == 'live':
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if resize:
                    frame = cv2.resize(frame, (resize[1], resize[0])) 

                depthMap = self.__processImage(frame, inputWidth, inputHeight, colormap)

                if show:
                    cv2.imshow('Depth Map', depthMap)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

    def __trainPerEpoch(self, 
                        trainLoader: Any, 
                        gradientClip: float) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Args:
            `trainLoader` (torch.utils.data.DataLoader): DataLoader for training dataset.
            `gradientClip` (float, optional): Gradient clipping value. Default is None.

        Returns:
            `Tuple[float, float]`: Average loss and accuracy for the epoch.
        """
        epochLoss = 0
        epochCorrect = 0
        epochTotal = 0
        
        try:
            self.train()
            with alive_bar(len(trainLoader)) as bar:
                for images, labels in trainLoader:
                    images, labels = self.__selectAugmentation((images, labels))
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self(images)
                    outputs = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)

                    loss = self.loss(outputs, labels)
                    loss.backward()
                    
                    if gradientClip is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), gradientClip)
                    
                    self.optimizer.step()
                    
                    epochLoss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    epochCorrect += (predicted == labels).sum().item()
                    epochTotal += labels.size(0)
                    
                    bar()
            
            epochAccuracy = epochCorrect / epochTotal if epochTotal > 0 else 0.0
        
            return epochLoss / len(trainLoader), epochAccuracy

    
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("WARNING: Ran out of memory during training. Consider reducing batch size or model size.")
            elif 'CUDA out of memory' in str(e):
                print("WARNING: CUDA out of memory during training. Consider reducing batch size or model size, or using smaller inputs.")
            else:
                raise e

    def __evaluatePerEpoch(self, 
                           validationLoader: Any) -> Tuple[float, float]:
        """
        Evaluate the model for one epoch.

        Args:
            `validationLoader` (torch.utils.data.DataLoader): DataLoader for validation dataset.

        Returns:
            `Tuple[float, float]`: Average loss and accuracy for the epoch.
        """
        epochLoss = 0
        epochCorrect = 0
        epochTotal = 0
        
        try:
            self.eval()
            with alive_bar(len(validationLoader)) as bar:
                with torch.no_grad():
                    for images, labels in validationLoader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self(images)
                        outputs = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)

                        loss = self.loss(outputs, labels)
                        
                        epochLoss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        epochCorrect += (predicted == labels).sum().item()
                        epochTotal += labels.size(0)
                        
                        bar()
            
            epochAccuracy = epochCorrect / epochTotal if epochTotal > 0 else 0.0
        
            return epochLoss / len(validationLoader), epochAccuracy
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("WARNING: Ran out of memory during evaluation. Consider reducing batch size or model size.")
            elif 'CUDA out of memory' in str(e):
                print("WARNING: CUDA out of memory during evaluation. Consider reducing batch size or model size, or using smaller inputs.")
            else:
                raise e

    def __buildEncoder(self, 
                    numLayers: int, 
                    kernelSize: int, 
                    numKernels: int, 
                    inputChannels: int) -> torch.nn.Sequential:
        """
        Build the encoder layers.

        Args:
            `numLayers` (int): Number of convolutional layers.
            `kernelSize` (int): Kernel size for convolutional layers.
            `numKernels` (int): Number of output channels for convolutional layers.
            `inputChannels` (int): Number of input channels for the first convolutional layer.

        Returns:
            `torch.nn.Sequential`: Sequential model representing the encoder layers.
        """
        encoderLayers = []
        for i in range(numLayers):
            encoderLayers.append(nn.Conv2d(inputChannels, numKernels[i], kernelSize, stride=2, padding=1))
            encoderLayers.append(nn.ReLU(inplace=True))
            encoderLayers.append(nn.BatchNorm2d(numKernels[i]))
            inputChannels = numKernels[i]

        return nn.Sequential(*encoderLayers)

    def __buildDecoder(self, 
                       numLayers: int, 
                       kernelSize: int, 
                       inputChannels: int, 
                       outputChannels: int) -> torch.nn.Sequential:
        """
        Build the decoder layers.

        Args:
            `numLayers` (int): Number of convolutional layers.
            `kernelSize` (int): Kernel size for convolutional layers.
            `numKernels` (int): Number of output channels for convolutional layers.
            `inputChannels` (int): Number of input channels for the first convolutional layer.

        Returns:
            `torch.nn.Sequential`: Sequential model representing the decoder layers.
        """
        decoderLayers = []
        for _ in range(numLayers - 1):
            decoderLayers.append(nn.ConvTranspose2d(inputChannels, inputChannels // 2, kernelSize, stride=2, padding=1, output_padding=1))
            decoderLayers.append(nn.ReLU(inplace=True))
            decoderLayers.append(nn.BatchNorm2d(inputChannels // 2))
            inputChannels //= 2

        decoderLayers.append(nn.ConvTranspose2d(inputChannels, outputChannels, kernelSize, stride=2, padding=1, output_padding=1))
        return nn.Sequential(*decoderLayers)

    def __processImage(self, 
                       image: Any, 
                       inputWidth: int, 
                       inputHeight: int, 
                       colormap: str) -> np.ndarray:
        """
        Preprocesses the input image, performs a forward pass through the model, and post-processes the depth map.

        Args:
            `image` (Any): The input image.
            `inputWidth` (int): The width of the input image.
            `inputHeight` (int): The height of the input image.
            `colormap` (str): The choice of colormap for visualization.

        Returns:
            np.ndarray: The processed depth map.
        """
        if isinstance(image, np.ndarray):
            imageTensor = self.__preprocessImage(image, inputWidth, inputHeight)
        elif isinstance(image, torch.Tensor):
            imageTensor = image.to(self.device)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            imageTensor = self.__preprocessImage(image, inputWidth, inputHeight)
        else:
            raise ValueError("Unsupported input image type. Supported types: numpy.ndarray, torch.Tensor, PIL.Image.Image")
        
        depthMap = self.forward(imageTensor)

        depthMap = depthMap.squeeze().cpu().detach().numpy()
        depthMap = self.__postprocessDepthmap(depthMap, colormap)

        return depthMap

    def __preprocessImage(self, 
                        image: Union[Image.Image, np.ndarray, torch.Tensor], 
                        inputWidth: int, 
                        inputHeight: int, 
                        mean: Tuple[float] = (0.485, 0.456, 0.406), 
                        std: Tuple[float] = (0.229, 0.224, 0.225), 
                        additionalTransforms: List[Any] = None,
                        resize: bool = True,
                        crop: bool = False,
                        augmentations: List[Any] = None) -> torch.Tensor:
        """
        Preprocesses the input image for model input.

        Args:
            image (Union[Image.Image, np.ndarray, torch.Tensor]): The input image.
            inputWidth (int): The width of the input image.
            inputHeight (int): The height of the input image.
            mean (Tuple[float], optional): Mean values for normalization. Defaults to (0.485, 0.456, 0.406).
            std (Tuple[float], optional): Standard deviation values for normalization. Defaults to (0.229, 0.224, 0.225).
            additionalTransforms (List[Any], optional): Additional transformations to be applied. Defaults to None.
            resize (bool, optional): Whether to resize the image. Defaults to True.
            crop (bool, optional): Whether to center crop the image. Defaults to False.
            augmentations (List[Any], optional): List of augmentations to apply. Defaults to None.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        if additionalTransforms is None:
            additionalTransforms = []

        if augmentations is None:
            augmentations = []

        transformList = []

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.permute(0, 3, 1, 2)
            elif image.dim() == 3:
                image = image.permute(2, 0, 1)
            image = transforms.ToPILImage()(image)
                    
        if resize:
            transformList.append(transforms.Resize((inputHeight, inputWidth), antialias=True))

        if crop:
            transformList.append(transforms.CenterCrop((inputHeight, inputWidth)))

        transformList.extend(augmentations)
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize(mean=mean, std=std))
        transformList.extend(additionalTransforms)

        transform = transforms.Compose(transformList)

        return transform(image).unsqueeze(0)

    def __postprocessDepthmap(self, 
                              depthMap: torch.Tensor, 
                              colormap: str) -> np.ndarray:
        """
        Post-processes the depth map.

        Args:
            `depthMap` (torch.Tensor): The depth map tensor.
            `colormap` (str): The choice of colormap. Supported values are 'colorized' and 'grayscale'.

        Returns:
            `np.ndarray`: The post-processed depth map.
        """
        depthMap = depthMap.cpu().detach().numpy()
        depthMapNormalized = (depthMap - depthMap.min()) / (depthMap.max() - depthMap.min())

        if self.minDepth is not None:
            depthMapNormalized = np.maximum(depthMapNormalized, self.minDepth)
        if self.maxDepth is not None:
            depthMapNormalized = np.minimum(depthMapNormalized, self.maxDepth)

        if colormap == 'colorized':
            depthMapVisualized = cv2.applyColorMap((depthMapNormalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        elif colormap == 'grayscale':
            depthMapVisualized = (depthMapNormalized * 255).astype(np.uint8)

        return depthMapVisualized
    
    def __cutMix(self, 
                batch: Tuple[torch.Tensor], 
                alpha: float = 0.1) -> Tuple[torch.Tensor]:
        """
        Apply CutMix augmentation to a batch of input images and labels.

        Args:
            batch (Tuple[torch.Tensor]): A tuple containing input images and their corresponding labels.
            alpha (float): The CutMix parameter controlling the extent of augmentation. Defaults to 0.1.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the augmented input images and labels.
        """
        images, labels = batch
        batchSize = images.size(0)
        indices = torch.randperm(batchSize)
        shuffledImages = images[indices]
        shuffledLabels = labels[indices]

        lambdaParam = np.random.beta(alpha, alpha)
        cutRatio = np.sqrt(1.0 - lambdaParam)

        cutWidth = int(images.size(2) * cutRatio)
        cutHeight = int(images.size(3) * cutRatio)

        centerX = np.random.randint(images.size(2))
        centerY = np.random.randint(images.size(3))

        boundBoxTopLeftX = np.clip(centerX - cutWidth // 2, 0, images.size(2))
        boundBoxTopLeftY = np.clip(centerY - cutHeight // 2, 0, images.size(3))
        boundBoxBottomRightX = np.clip(centerX + cutWidth // 2, 0, images.size(2))
        boundBoxBottomRightY = np.clip(centerY + cutHeight // 2, 0, images.size(3))

        images[:, :, boundBoxTopLeftX:boundBoxBottomRightX, boundBoxTopLeftY:boundBoxBottomRightY] = shuffledImages[:, :, boundBoxTopLeftX:boundBoxBottomRightX, boundBoxTopLeftY:boundBoxBottomRightY]
        labels = (1 - lambdaParam) * labels + lambdaParam * shuffledLabels

        return images, labels

    def __mixUp(self, 
                batch: Tuple[torch.Tensor], 
                alpha: float = 0.1) -> Tuple[torch.Tensor]:
        """
        Apply MixUp augmentation to a batch of input images and labels.

        Args:
            batch (Tuple[torch.Tensor]): A tuple containing input images and their corresponding labels.
            alpha (float): The MixUp parameter controlling the extent of augmentation. Defaults to 0.1.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the augmented input images and labels.
        """
        images, labels = batch
        batchSize = images.size(0)
        indices = torch.randperm(batchSize)
        shuffledImages = images[indices]
        shuffledLabels = labels[indices]

        lambdaParam = np.random.beta(alpha, alpha)
        images = lambdaParam * images + (1 - lambdaParam) * shuffledImages
        labels = lambdaParam * labels + (1 - lambdaParam) * shuffledLabels

        return images, labels

    def __selectAugmentation(self, 
                            batch: Tuple[torch.Tensor], 
                            alpha: float = 1.0) -> Tuple[torch.Tensor]:
        """
        Randomly selects an augmentation technique for the given batch of images and labels.

        Args:
            `batch` (Tuple[torch.Tensor]): A tuple containing images and corresponding labels.
            `alpha` (float, optional): The alpha parameter for MixUp or CutMix augmentation. Defaults to 1.0.

        Returns:
            `Tuple[torch.Tensor]`: A tuple containing augmented images and labels based on 
                the selected augmentation technique.
        """
        REGULAR = 0.6
        CUTMIX = 0.2
        MIXUP = 0.2

        images, labels = batch

        rand = random.random() 

        if rand < REGULAR:
            return images, labels
        elif rand < REGULAR + CUTMIX:
            return self.__cutMix((images, labels), alpha)
        else:
            return self.__mixUp((images, labels), alpha)
