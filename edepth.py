import os
import time

import cv2
from PIL import Image

from typing import Any, Optional, Union
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim.lr_scheduler  as Scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import swats

from alive_progress import alive_bar

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class edepth(nn.Module):
    """
    Neural network model for depth estimation using an encoder-decoder architecture.

    This class defines a neural network (`edepth`) designed specifically for depth estimation tasks.
    It utilizes an encoder-decoder architecture, commonly used in computer vision tasks, to predict depth maps
    from input images.

    The model consists of the following components:
    - Encoder: Extracts hierarchical features from input images to capture contextual information.
    - Fully Connected Layers: Adapt the output of the encoder for input into the decoder and vice versa.
    - Decoder: Upsamples the encoded features to generate high-resolution depth maps.

    Attributes
    ----------
    __author__ : Ehsan Asgharzadeh
    __copyright__ : Copyright (C) 2024 edepth
    __license__ : https://ehsanasgharzde.ir
    __contact__ : https://ehsanasgharzadeh.asg@gmail.com
    __date__ : 2024 Jun 22
    __version__ : 1.0.1

    Parameters
    ----------
    optimizer : Any, optional
        Optimizer for model training (default is None, which initializes SWATS optimizer with lr=0.0001).
    activation : Any, optional
        Activation function applied to model layers (default is None, which initializes ReLU activation).
    loss : Any, optional
        Loss function used for training (default is None, which initializes Mean Squared Error loss).
    scheduler : Any, optional
        Learning rate scheduler (default is None, which initializes ReduceLROnPlateau scheduler with patience=100 and factor=0.5).
    dropout : Any, optional
        Dropout probability for regularization (default is None, which initializes no dropout).

    Methods
    -------
    __init__(self, optimizer=None, activation=None, loss=None, scheduler=None, dropout=None, inputChannels=3, growthRate=32, neurons=1024):
        Initializes the edepth model with provided or default components.

    How to Use
    ----------
    1. Initialization:
    ```python
    model = edepth(optimizer=optimizerInstance, activation=nn.ReLU(), loss=nn.MSELoss(), scheduler=schedulerInstance)
    ```
    Initialize an instance of `edepth` with optional components such as optimizer, activation function,
    loss function, and learning rate scheduler.

    2. Loading Pre-trained Model:
    ```python
    model.eload(filePath='path_to_your_model.pt')
    ```
    Load a pre-trained model from a file using the `eload` method.

    3. Forward Pass (Depth Estimation):
    ```python
    inputImagePath = 'path_to_your_image.jpg'  # Example input image path
    model.egenerate(source='image', inputFilePath=inputImagePath, show=True, save=False)
    ```
    Generate a depth map from an input image using the `egenerate` method with options to display (`show=True`)
    or save (`save=True`) the depth map.

    4. Training:
    ```python
    trainDataset = YourDatasetHere(...)  # Define your training dataset
    model.etrain(trainset=trainDataset, epochs=numEpochs, batchSize=batchSize)
    ```
    Train the model using a DataLoader with batches of training data. Use the `etrain` method to
    handle the training loop, loss calculation, and optimization.

    Notes
    -----
    - The encoder is instantiated with specified input channels, growth rate, and number of layers.
    - The fully connected layers adapt the encoder output for decoder input and vice versa.
    - Optional components like optimizer, activation function, loss function, scheduler, and dropout
    can be provided during initialization.
    - If not provided, default implementations are used for these components.
    - This class inherits from `nn.Module`, making it compatible with PyTorch's training and evaluation APIs.
    """
    __author__ = "Ehsan Asgharzadeh"
    __copyright__ = "Copyright (C) 2024 edepth"
    __license__ = "https://ehsanasgharzde.ir"
    __contact__ = "https://ehsanasgharzadeh.asg@gmail.com"
    __date__ = "2024 Jun 22"
    __version__ = "1.0.1"

    class _DenseBlock(nn.Module):
        """
        A dense block used in the edepth class.

        This block consists of two convolutional layers with batch normalization,
        designed to process input data and concatenate the output with the input.

        Parameters
        ----------
        inputChannels : int
            The number of input channels.
        growthRate : int
            The growth rate of the network, defining the number of output channels
            for each convolutional layer.
        device : torch.device
            The device on which the model's parameters should be initialized (e.g., 'cuda' or 'cpu').

        Attributes
        ----------
        batchNormI : nn.BatchNorm2d
            Batch normalization layer for the input channels.
        convolutionI : nn.Conv2d
            First convolutional layer with a kernel size of 1.
        batchNormII : nn.BatchNorm2d
            Batch normalization layer for the growth rate channels.
        convolutionII : nn.Conv2d
            Second convolutional layer with a kernel size of 3 and padding of 1.

        Methods
        -------
        forward(x)
            Defines the forward pass of the dense block.
        """
        def __init__(self, inputChannels, growthRate, device):
            super(edepth._DenseBlock, self).__init__()

            self.to(device)
            self.batchNormI = nn.BatchNorm2d(inputChannels, device=device)
            self.convolutionI = nn.Conv2d(inputChannels, growthRate, kernel_size=1, device=device)
            self.batchNormII = nn.BatchNorm2d(growthRate, device=device)
            self.convolutionII = nn.Conv2d(growthRate, growthRate, kernel_size=3, padding=1, device=device)


        def forward(self, x):
            output = self.convolutionI(self.batchNormI(x))
            output = self.convolutionII(self.batchNormII(output))
            output = torch.cat([x, output], 1)
            return output

    class _TransitionLayer(nn.Module):
        """
        A translation layer used in the edepth class.

        This layer consists of a convolutional layer followed by a max-pooling layer,
        designed to reduce the dimensions of the input data while changing the number
        of channels.

        Parameters
        ----------
        inputChannels : int
            The number of input channels.
        outputChannels : int
            The number of output channels.
        device : torch.device
            The device on which the model's parameters should be initialized (e.g., 'cuda' or 'cpu').

        Attributes
        ----------
        convolution : nn.Conv2d
            Convolutional layer with a kernel size of 1.
        pool : nn.MaxPool2d
            Max-pooling layer with a kernel size of 2 and stride of 2.

        Methods
        -------
        forward(x)
            Defines the forward pass of the transition layer.
        """
        def __init__(self, inputChannels, outputChannels, device):
            super(edepth._TransitionLayer, self).__init__()

            self.to(device)
            self.convolution = nn.Conv2d(inputChannels, outputChannels, kernel_size=1, device=device)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            output = self.convolution(x)
            output = self.pool(output)
            return output

    class _Encoder(nn.Module):
        """
        An encoder module used in the edepth class.

        This encoder consists of an initial convolutional layer followed by a pooling layer,
        a series of dense blocks, and transition layers. It processes the input data through
        these layers to reduce its dimensions while extracting features.

        Parameters
        ----------
        inputChannels : int
            The number of input channels.
        growthRate : int
            The growth rate of the network, defining the number of output channels for each dense block.
        numLayers : int
            The number of dense blocks in the encoder.
        device : torch.device
            The device on which the model's parameters should be initialized (e.g., 'cuda' or 'cpu').

        Attributes
        ----------
        initialConvolution : nn.Conv2d
            Initial convolutional layer with a kernel size of 5 and padding of 2.
        pool : nn.MaxPool2d
            Max-pooling layer with a kernel size of 2 and stride of 2.
        denseBlocks : nn.ModuleList
            List of dense blocks.
        transitionLayers : nn.ModuleList
            List of transition layers.
        outputChannels : int
            The number of output channels after the final dense block and transition layer.

        Methods
        -------
        forward(x)
            Defines the forward pass of the encoder.
        """
        def __init__(self, inputChannels, growthRate, numLayers, device):
            super(edepth._Encoder, self).__init__()

            self.to(device)
            self.initialConvolution = nn.Conv2d(inputChannels, growthRate, kernel_size=5, padding=2, device=device)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.denseBlocks = nn.ModuleList()
            self.transitionLayers = nn.ModuleList()

            
            numChannels = growthRate
            for i in range(numLayers):
                self.denseBlocks.append(edepth._DenseBlock(numChannels, growthRate, device=device))
                numChannels += growthRate
                if i != numLayers - 1:
                    self.transitionLayers.append(edepth._TransitionLayer(numChannels, numChannels // 2, device=device))
                    numChannels = numChannels // 2
            self.outputChannels = numChannels // 2

        def forward(self, x):
            output = self.initialConvolution(x)
            output = self.pool(output)
            for i, block in enumerate(self.denseBlocks):
                output = block(output)
                if i != len(self.denseBlocks) - 1:
                    output = self.transitionLayers[i](output)
            return output

    class _Decoder(nn.Module):
        """
        A decoder module used in the edepth class.

        This decoder consists of a series of upsampling layers followed by convolutional layers.
        It processes the input data through these layers to upsample and reduce the number of channels,
        ultimately producing a single-channel output.

        Parameters
        ----------
        inputChannels : int
            The number of input channels.
        device : torch.device
            The device on which the model's parameters should be initialized (e.g., 'cuda' or 'cpu').

        Attributes
        ----------
        upSampleI : nn.Upsample
            First upsampling layer with a scale factor of 2.
        convI : nn.Conv2d
            First convolutional layer with a kernel size of 3 and padding of 1.
        upSampleII : nn.Upsample
            Second upsampling layer with a scale factor of 2.
        convII : nn.Conv2d
            Second convolutional layer with a kernel size of 3 and padding of 1.
        upSampleIII : nn.Upsample
            Third upsampling layer with a scale factor of 2.
        convIII : nn.Conv2d
            Third convolutional layer with a kernel size of 3 and padding of 1.
        upSampleIV : nn.Upsample
            Fourth upsampling layer with a scale factor of 2.
        convIV : nn.Conv2d
            Fourth convolutional layer with a kernel size of 3 and padding of 1.

        Methods
        -------
        forward(x)
            Defines the forward pass of the decoder.
        """
        def __init__(self, inputChannels, device):
            super(edepth._Decoder, self).__init__()
            self.upSampleI = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.convI = nn.Conv2d(inputChannels, inputChannels // 2, kernel_size=3, padding=1, device=device)
            
            self.upSampleII = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.convII = nn.Conv2d(inputChannels // 2, inputChannels // 4, kernel_size=3, padding=1, device=device)
            
            self.upSampleIII = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.convIII = nn.Conv2d(inputChannels // 4, inputChannels // 8, kernel_size=3, padding=1, device=device)
            
            self.upSampleIV = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.convIV = nn.Conv2d(inputChannels // 8, 1, kernel_size=3, padding=1, device=device)

        def forward(self, x):
            x = self.upSampleI(x)
            x = nn.functional.relu(self.convI(x))
            
            x = self.upSampleII(x)
            x = nn.functional.relu(self.convII(x))
            
            x = self.upSampleIII(x)
            x = nn.functional.relu(self.convIII(x))
            
            x = self.upSampleIV(x)
            x = torch.tanh(self.convIV(x))
            
            return x
        
    def __init__(self,  optimizer: Any = None, activation: Any = None, loss: Any = None, scheduler: Any = None, dropout: Any = None, inputChannels: int = 3, growthRate: int = 32, neurons: int = 1024) -> None:
        super(edepth, self).__init__()
        self.device, self.To = self.__getDevice()

        self.encoder = edepth._Encoder(inputChannels, growthRate, 4, device=self.device)

        encoderOutSize = self.encoder.outputChannels * 2 * 14 * 14
        self.fullyConnectedI = nn.Linear(encoderOutSize, neurons, device=self.device)
        self.fullyConnectedII = nn.Linear(neurons, encoderOutSize, device=self.device)

        self.decoder = edepth._Decoder(growthRate * 2, self.device)

        self.optimizer = optimizer if optimizer is not None else swats.SWATS(self.parameters(), lr=0.0001)
        self.activation = activation if activation is not None else nn.ReLU()
        self.loss = loss if loss is not None else nn.MSELoss()
        self.scheduler = scheduler if scheduler is not None else torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=100,  factor=0.5)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
 
    def __del__(self) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            del self.encoder
            del self.decoder
        except:
            pass

    def __str__(self) -> str:
        return f"<class 'edepth' at {hex(id(self))}>"

    def __repr__(self) -> str:
        return f"Author: Ehsan Asgharzadeh <https://ehsanasgharzde@gmail.com>\nLicense: https://ehsanasgharzde.ir"
   
    def __class__(self) -> str:
        return "<class 'edepth'>"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input tensor must have shape (batchSize, channels, height, width).")
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fullyConnectedI(x))
        x = self.dropout(x)
        x = self.activation(self.fullyConnectedII(x))
        x = x.view(x.size(0), self.encoder.outputChannels * 2, 14, 14)
        x = self.decoder(x)
        return x
    
    def eload(self, filePath: str) -> None:
        """
        Load the model state from a file.

        This method loads the model's state dictionary from a specified file path.
        It handles file not found and runtime errors that might occur during the process.

        Parameters
        ----------
        filePath : str
            The path to the file from which to load the model state.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If there is an error while loading the model state.

        Examples
        --------
        >>> model.eload('path/to/model.pth')
        Model `model.pth` loaded successfully.

        >>> model.eload('nonexistent/file.pth')
        Traceback (most recent call last):
            ...
        FileNotFoundError: FileNotFoundError | File `nonexistent/file.pth` not found.
        """
        try:
            self.load_state_dict(torch.load(filePath, map_location=self.device))
            fileName = filePath.replace('\\', '/').split('/')[-1]
            print(f"Model '{fileName}' loaded successfully.")

        except FileNotFoundError:
            raise FileNotFoundError(f"FileNotFoundError | File '{filePath}' not found.")
        except RuntimeError as e:
            raise RuntimeError(f"RuntimeError | Error while loading model: \n{str(e)}")

    def etrain(self, trainset: Dataset, validationset: Optional[Dataset] = None, epochs: int = 10, batchSize: int = 32, shuffle: bool = True, validationMetrics: Optional[Dict[str, Any]] = None, gradientClip: Optional[float] = None, logInterval: Optional[int] = None, earlyStoppingPatience: int = 10, checkpointPath: Optional[str] = None, saveBestOnly: bool = True) -> None:
        """
        Train the model on the provided datasets.

        This method trains the model on the `trainset` dataset for `epochs` epochs. Optionally, it can validate the model on the `validationset` dataset,
        save checkpoints of the best model based on validation loss, perform early stopping, and log training progress.

        Parameters
        ----------
        trainset : Dataset
            The training dataset.
        validationset : Optional[Dataset], optional
            The validation dataset (default is None).
        epochs : int, optional
            The number of epochs to train the model (default is 10).
        batchSize : int, optional
            The batch size used for training and validation (default is 32).
        shuffle : bool, optional
            Whether to shuffle the training data at the beginning of each epoch (default is True).
        validationMetrics : Optional[Dict[str, Any]], optional
            Additional metrics to evaluate on the validation set (default is None).
        gradientClip : Optional[float], optional
            Maximum norm of gradients for gradient clipping during training (default is None).
        logInterval : Optional[int], optional
            Interval for logging training progress in number of epochs (default is None).
        earlyStoppingPatience : int, optional
            Number of epochs to wait for improvement in validation loss before triggering early stopping (default is 10).
        checkpointPath : Optional[str], optional
            Path to save checkpoints of the model with the best validation loss (default is None).
        saveBestOnly : bool, optional
            Whether to save only the best model based on validation loss (default is True).

        Returns
        -------
        validationLoss : float or None
            The final validation loss after training, or None if `validationset` is None.
        validationAccuracy : float or None
            The final validation accuracy after training, or None if `validationset` is None.
        trainLoss : float
            The final training loss after training.
        trainAccuracy : float
            The final training accuracy after training.

        Raises
        ------
        RuntimeError
            If there is an error during the training process.

        Examples
        --------
        >>> model.etrain(train_dataset, validation_dataset, epochs=20, batchSize=64, earlyStoppingPatience=5)
        edepth Parameters: 10M
        Epoch: 01 | Epoch Time: 1m 30s
            Train Loss: 0.345 | Train Accuracy: 87.3%
            Validation Loss: 0.212 | Validation Accuracy: 91.5%
        Epoch: 02 | Epoch Time: 1m 28s
            Train Loss: 0.278 | Train Accuracy: 89.7%
            Validation Loss: 0.198 | Validation Accuracy: 92.1%
        ...

        Notes
        -----
        - If `validationMetrics` is provided, these metrics are printed after each validation evaluation.
        - If `saveBestOnly` is True, only the model with the best validation loss is saved to `checkpointPath`.
        - The learning rate scheduler, if specified in the model, is updated after each epoch.
        """
        try:
            trainLoader = DataLoader(trainset, batch_size=batchSize, shuffle=shuffle)
            validationLoader = DataLoader(validationset, batch_size=batchSize, shuffle=False) if validationset is not None else None
            bestValidationLoss = float('inf')
            epochsWithNoImprove = 0
            
            print('edepth Parammeters: ', f"{sum(parameters.numel() for parameters in self.parameters(recurse=True)) / 1e6:.0f}M")
            for epoch in range(1, epochs + 1):
                startTime = time.monotonic()
                trainLoss, trainAccuracy = self.__trainPerEpoch(trainLoader, gradientClip)
                validationLoss, validationAccuracy = self.__evaluatePerEpoch(validationLoader) if validationLoader is not None else (None, None)

                if validationLoss is not None and validationLoss < bestValidationLoss:
                    bestValidationLoss = validationLoss
                    if checkpointPath:
                        torch.save(self.state_dict(), os.path.join(checkpointPath, f"ep-{epoch}-valacc-{validationAccuracy}.pt"))

                endTime = time.monotonic()
                epochMinutes, epochSeconds = divmod(endTime - startTime, 60)

                print(f'Epoch: {epoch:02} | Epoch Time: {epochMinutes:.0f}m {epochSeconds:.0f}s') 
                print(f'\tTrain Loss: {trainLoss} | Train Accuracy: {trainAccuracy * 100}%') 
                if validationLoss is not None:
                    print(f'\tValidation Loss: {validationLoss} | Validation Accuracy: {validationAccuracy * 100}%')

                if validationMetrics is not None:
                    for metricName, metricValue in validationMetrics.items():
                        print(f'\t{metricName}: {metricValue:.3f}')

                if saveBestOnly and validationLoss is not None and validationLoss >= bestValidationLoss:
                    epochsWithNoImprove += 1
                    if epochsWithNoImprove >= earlyStoppingPatience:
                        print(f"Early stopping triggered with {epochsWithNoImprove} epoch with no improvements.")
                        break
                else:
                    epochsWithNoImprove = 0

                if self.scheduler:
                    if isinstance(self.scheduler, _LRScheduler):
                        if isinstance(self.scheduler, Scheduler.ReduceLROnPlateau):
                            self.scheduler.step(validationLoss)
                        else:
                            self.scheduler.step()

                if logInterval is not None and epoch % logInterval == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {trainLoss:.4f}")

            return validationLoss, validationAccuracy, trainLoss, trainAccuracy

        except Exception as e:
            raise RuntimeError(f"RuntimeError | Error during training:\n{str(e)}")

    def egenerate(self, source: str, inputFilePath: str = 'input', show: bool = False, save: bool = True, outputDir: str = 'output', outputFormat: str = 'jpg', outputFilename: str = 'depthmap', colormap: str = 'colorized', frameRange: tuple = None, resize: tuple = (224, 224), minDepth: float = None, maxDepth: float = None) -> None:
        """
        Generate depth maps from images, videos, or live camera feed.

        This method processes input data from various sources (image, video, live feed) to generate depth maps using the model.

        Parameters
        ----------
        source : str
            The source of input data. Choices are 'image' for single image file, 'video' for video file, or 'live' for live camera feed.
        inputFilePath : str
            The path to the input file (image or video) or camera index (for live feed).
        show : bool, optional
            Whether to display the generated depth map (default is False).
        save : bool, optional
            Whether to save the generated depth map to disk (default is True).
        outputDir : str, optional
            The directory to save the output depth maps (default is 'depthmaps').
        outputFormat : str, optional
            The format of the output depth map file ('jpg', 'png', etc.) if saving (default is 'jpg').
        outputFilename : str, optional
            The base filename for the saved depth map (default is 'depthmap').
        colormap : str, optional
            The colormap to apply to the depth map ('grayscale', 'colorized') (default is 'colorized').
        frameRange : tuple, optional
            Range of frames to process from the video (start frame index, end frame index) (default is None, processes all frames).
        resize : tuple, optional
            The dimensions to resize the input frames (height, width) (default is (224, 224)).
        minDepth : float, optional
            Minimum depth value for depth map visualization (default is None, auto-scaled).
        maxDepth : float, optional
            Maximum depth value for depth map visualization (default is None, auto-scaled).

        Raises:
        ------
        ValueError
            If an invalid `source` is provided.
        FileNotFoundError
            If the specified `inputFilePath` does not exist.

        Notes:
        ------
        - When `source` is 'image', it reads the image from `inputFilePath`, generates a depth map using the internal 'edepth' model,
          and optionally displays and/or saves the depth map.
        - When `source` is 'video', it reads the video from `inputFilePath`, processes each frame to generate depth maps,
          and optionally displays and/or saves each frame's depth map.
        - When `source` is 'live', it captures frames from the live camera feed, generates depth maps in real-time,
          and optionally displays and/or saves each frame's depth map.
        - The depth maps can be visualized either in grayscale or colorized format based on the `colormap` parameter.

        Example:
        --------
        ```python
        model = edepth()
        model.eload('path_to_model.pt')  # Load a pre-trained model
        
        # Generate depth map from an image
        model.egenerate(source='image', inputFilePath='path_to_image.jpg', show=True, save=True)

        # Generate depth maps from a video and save them
        model.egenerate(source='video', inputFilePath='path_to_video.mp4', show=False, save=True)

        # Generate and display depth map from live camera feed
        model.egenerate(source='live', inputFilePath='', show=True, save=False)
        ```
        """
        if save and not os.path.exists(outputDir):
            os.makedirs(outputDir)

        if source == 'image':
            image = cv2.imread(inputFilePath)
            depthMap = self.__processImage(image, resize[1], resize[0], colormap, minDepth, maxDepth)
            depthMapResized = cv2.resize(depthMap, (image.shape[1], image.shape[0]))

            if show:
                depthMapToShow = cv2.resize(depthMap, (image.shape[1] // 2, image.shape[0] // 2))
                imageToShow = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
                cv2.imshow('edepth - image', np.hstack((imageToShow, depthMapToShow)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save:
                outputPath = os.path.join(outputDir, 'colorized', f'{outputFilename}.{outputFormat}') if colormap == 'colorized' else os.path.join(outputDir, 'grayscale', f'{outputFilename}_pred.{outputFormat}')
                cv2.imwrite(outputPath, depthMapResized)
        
        elif source == 'video':
            cap = cv2.VideoCapture(inputFilePath)
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = None
            if save:
                outputPath = os.path.join(outputDir, 'video', f'{outputFilename}_vidPred.mp4')
                out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frameWidth, frameHeight))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frameRange and cap.get(cv2.CAP_PROP_POS_FRAMES) < frameRange[0]:
                    continue
                if frameRange and cap.get(cv2.CAP_PROP_POS_FRAMES) > frameRange[1]:
                    break
                
                depthMap = self.__processImage(frame, resize[1], resize[0], colormap, minDepth, maxDepth)
                depthMapResized = cv2.resize(depthMap, (frameWidth, frameHeight))

                if show:
                    depthMapToShow = cv2.resize(depthMap, (frameWidth // 2, frameHeight // 2))
                    frameToShow = cv2.resize(frame, (frameWidth // 2, frameHeight // 2))
                    cv2.imshow('edepth - video', np.hstack((frameToShow, depthMapToShow)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save:
                    out.write(depthMapResized)

            cap.release()
            if save:
                out.release()
            cv2.destroyAllWindows()
        
        elif source == 'live':
            cap = cv2.VideoCapture(0)
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = None
            if save:
                outputPath = os.path.join(outputDir, 'video', f'{outputFilename}_feedPred.mp4')
                out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frameWidth * 2, frameHeight))

            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if not ret:
                    break

                depthMap = self.__processImage(frame, resize[1], resize[0], colormap, minDepth, maxDepth)
                depthMapResized = cv2.resize(depthMap, (frameWidth, frameHeight))
                combinedFrame = np.hstack((frame, depthMapResized))

                if show:
                    cv2.imshow('edepth - live feed', combinedFrame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save:
                    out.write(combinedFrame)

            cap.release()
            cv2.destroyAllWindows()

    def __trainPerEpoch(self, trainLoader: Any, gradientClip: float) -> Tuple[float, float]:
        """
        Perform training for one epoch.

        This method executes one epoch of training using the provided data loader `trainLoader`. It computes the loss,
        performs backpropagation, and updates the model parameters.

        Parameters
        ----------
        trainLoader : Any
            The data loader containing batches of training data.
        gradientClip : float
            The maximum norm value for gradient clipping during backpropagation.

        Returns
        -------
        float
            The average loss per batch for the epoch.
        float
            The accuracy of the model for the epoch.

        Raises
        ------
        RuntimeError
            If there is a CUDA memory issue during training.

        Notes
        -----
        - This method assumes the model (`self`) is in training mode (`self.train()`).
        - It uses the optimizer (`self.optimizer`) initialized outside this method.
        - Progress during training is visualized with an alive bar.
        - Gradient clipping is applied if `gradientClip` is provided.
        - CUDA memory warnings are printed if memory issues occur during training.
        """
        epochLoss = 0
        epochCorrect = 0
        epochTotal = 0
        
        try:
            self.train()
            with alive_bar(len(trainLoader)) as bar:
                for images, labels in trainLoader:
                    images, labels = images.to(self.To), labels.to(self.To)
                    
                    self.optimizer.zero_grad()
                    outputs = self(images)

                    loss = self.loss(outputs, labels)
                    loss.backward()
                    
                    if gradientClip is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), gradientClip)
                    
                    self.optimizer.step()
                    
                    epochLoss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correctPixels = (predicted == labels).sum().item()
                    totalPixels = labels.numel()

                    epochCorrect += correctPixels
                    epochTotal += totalPixels
                    
                    bar()
            
            epochAccuracy = correctPixels / totalPixels if totalPixels > 0 else 0.0
        
            return epochLoss / len(trainLoader), epochAccuracy

    
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("WARNING: Ran out of memory during training. Consider reducing batch size or model size.")
            elif 'CUDA out of memory' in str(e):
                print("WARNING: CUDA out of memory during training. Consider reducing batch size or model size, or using smaller inputs.")
            else:
                raise e

    def __evaluatePerEpoch(self, validationLoader: Any) -> Tuple[float, float]:
        """
        Perform evaluation for one epoch.

        This method evaluates the model performance on a validation set using the provided data loader `validationLoader`.
        It computes the average loss and accuracy for the entire validation set.

        Parameters
        ----------
        validationLoader : Any
            The data loader containing batches of validation data.

        Returns
        -------
        float
            The average loss per batch for the epoch.
        float
            The accuracy of the model for the epoch.

        Raises
        ------
        RuntimeError
            If there is a CUDA memory issue during evaluation.

        Notes
        -----
        - This method assumes the model (`self`) is in evaluation mode (`self.eval()`).
        - It does not perform gradient computation, utilizing `torch.no_grad()` context.
        - Progress during evaluation is visualized with an alive bar.
        - CUDA memory warnings are printed if memory issues occur during evaluation.
        """
        epochLoss = 0
        epochCorrect = 0
        epochTotal = 0
        
        try:
            self.eval()
            with alive_bar(len(validationLoader)) as bar:
                with torch.no_grad():
                    for images, labels in validationLoader:
                        images, labels = images.to(self.To), labels.to(self.To)
                        outputs = self(images)

                        loss = self.loss(outputs, labels)
                        
                        epochLoss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        correctPixels = (predicted == labels).sum().item()
                        totalPixels = labels.numel()

                        epochCorrect += correctPixels
                        epochTotal += totalPixels     

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

    def __processImage(self, image: Any, inputWidth: int, inputHeight: int, colormap: str, minDepth: float = None, maxDepth: float = None) -> np.ndarray:
        """
        Process an image to generate a depth map using the model.

        This method takes an input image, preprocesses it, computes the depth map using the model, and post-processes
        the depth map for visualization.

        Parameters
        ----------
        image : Any
            The input image data, expected in a format compatible with Torch tensors.
        inputWidth : int
            The width to resize the input image before processing.
        inputHeight : int
            The height to resize the input image before processing.
        colormap : str
            The colormap to apply to the depth map visualization ('grayscale', 'jet', etc.).
        minDepth : float, optional
            Minimum depth value for depth map visualization (default is None, auto-scaled).
        maxDepth : float, optional
            Maximum depth value for depth map visualization (default is None, auto-scaled).

        Returns
        -------
        np.ndarray
            The generated depth map as a NumPy array.

        Notes
        -----
        - This method assumes the model (`self`) is capable of processing Torch tensors.
        - The input image is resized to `(inputWidth, inputHeight)` before being fed into the model.
        - The depth map is computed using the model's `forward` method.
        - Postprocessing includes converting the depth map to a NumPy array and applying colormap visualization.
        """
        image = torch.from_numpy(image).to(self.To)
        image = self.__preprocessImage(image, inputWidth, inputHeight)
        depthMap = self.forward(image)

        depthMap = depthMap.squeeze().cpu().detach().numpy()
        depthMap = self.__postprocessDepthmap(depthMap, colormap, minDepth, maxDepth)
        
        return depthMap

    def __preprocessImage(self, image: Union[Image.Image, np.ndarray, torch.Tensor], inputWidth: int, inputHeight: int, mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225), additionalTransforms: List[Any] = None, resize: bool = True, crop: bool = False, augmentations: List[Any] = None) -> torch.Tensor:
        """
        Process an image to generate a depth map using the model.

        This method takes an input image, preprocesses it, computes the depth map using the model, and post-processes
        the depth map for visualization.

        Parameters
        ----------
        image : Any
            The input image data, expected in a format compatible with Torch tensors.
        inputWidth : int
            The width to resize the input image before processing.
        inputHeight : int
            The height to resize the input image before processing.
        colormap : str
            The colormap to apply to the depth map visualization ('grayscale', 'jet', etc.).
        minDepth : float, optional
            Minimum depth value for depth map visualization (default is None, auto-scaled).
        maxDepth : float, optional
            Maximum depth value for depth map visualization (default is None, auto-scaled).

        Returns
        -------
        np.ndarray
            The generated depth map as a NumPy array.

        Notes
        -----
        - This method assumes the model (`self`) is capable of processing Torch tensors.
        - The input image is resized to `(inputWidth, inputHeight)` before being fed into the model.
        - The depth map is computed using the model's `forward` method.
        - Postprocessing includes converting the depth map to a NumPy array and applying colormap visualization.
        """
        if additionalTransforms is None:
            additionalTransforms = []

        if augmentations is None:
            augmentations = []

        transformList = []

        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
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

        return transform(image).unsqueeze(0).to(self.To)

    def __postprocessDepthmap(self, depthMap: torch.Tensor | np.ndarray, colormap: str = 'grayscale', minDepth: float = None, maxDepth: float = None) -> np.ndarray:
        """
        Postprocess a depth map for visualization.

        This method normalizes the input depth map, applies optional clipping to minDepth and maxDepth values,
        and applies a colormap for visualization.

        Parameters
        ----------
        depthMap : torch.Tensor or np.ndarray
            The input depth map to be postprocessed.
        colormap : str, optional
            The colormap to apply to the depth map visualization ('grayscale', 'colorized', etc.) (default is 'grayscale').
        minDepth : float, optional
            Minimum depth value for depth map visualization (default is None, no clipping).
        maxDepth : float, optional
            Maximum depth value for depth map visualization (default is None, no clipping).

        Returns
        -------
        np.ndarray
            The postprocessed depth map as a NumPy array suitable for visualization.

        Notes
        -----
        - If `depthMap` is a Torch tensor, it is converted to a NumPy array using `.detach().numpy()`.
        - Depth values are normalized between 0 and 1.
        - Clipping to `minDepth` and `maxDepth` values is applied if provided.
        - Colormap choices include 'grayscale' (default) and 'colorized' (using the 'inferno' colormap).
        """
        if isinstance(depthMap, torch.Tensor):
            depthMap = depthMap.detach().numpy()
        depthMapNormalized = (depthMap - depthMap.min()) / (depthMap.max() - depthMap.min())

        if minDepth is not None:
            depthMapNormalized = np.maximum(depthMapNormalized, minDepth)
        if maxDepth is not None:
            depthMapNormalized = np.minimum(depthMapNormalized, maxDepth)

        if colormap == 'colorized':
            depthMapVisualized = cv2.applyColorMap((depthMapNormalized * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
        elif colormap == 'grayscale':
            depthMapVisualized = cv2.applyColorMap((depthMapNormalized * 255.0).astype(np.uint8), cv2.COLORMAP_BONE)

        return depthMapVisualized
    
    def __getDevice(self) -> torch.DeviceObjType:
        """
        Determine and return the appropriate device for computation.

        This method checks the availability of CUDA-enabled GPUs and MPS (Multi-Process Service), prioritizing GPU if
        available. If neither CUDA GPU nor MPS is available, it defaults to CPU.

        Returns
        -------
        torch.DeviceObjType
            The torch device object representing the selected device ('cuda', 'mps', or 'cpu').

        Notes
        -----
        - This method assumes the presence of Torch and checks for CUDA availability using `torch.cuda.is_available()`.
        - If CUDA is available, the method moves the model (`self`) to GPU ('cuda') and returns 'cuda'.
        - If MPS is available, it returns 'mps'.
        - If neither CUDA GPU nor MPS is available, it defaults to CPU and returns 'cpu'.
        """
        if torch.cuda.is_available():
            self.cuda()
            return torch.device('cuda'), 'cuda'
        self.cpu()        
        if torch.backends.mps.is_available():
            return torch.device('mps'), 'mps'

        return torch.device('cpu'), 'cpu'