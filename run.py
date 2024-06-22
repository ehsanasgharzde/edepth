import os
import time

from edepth import edepth
from utilities import Dataset, Sample

from sklearn.model_selection import train_test_split as trainTestSplit
import pandas as pd

    
# datasetDir = os.path.join(os.getcwd(), 'dataset/')
checkpointDir = os.path.join(os.getcwd(), 'checkpoints/')
inputDir = os.path.join(os.getcwd(), 'input/')
outputDir = os.path.join(os.getcwd(), 'output/')

# dataframe = pd.read_csv(datasetDir)
# train, validate = trainTestSplit(dataframe, test_size=0.2, random_state=42)
# trainset = Dataset(train, 224, 224)
# validationset = Dataset(validate, 224, 224)

inputs = Sample(inputDir)


growthRate = 32
neurons = 512
model = edepth(growthRate=growthRate,  neurons=neurons)

if __name__ == "__main__":
    # model.etrain(trainset=trainset, validationset=validationset, epochs=1000, batchSize=16, earlyStoppingPatience=100, gradientClip=2.0, checkpointPath=checkpointDir)
    model.eload(os.path.join(checkpointDir, 'ep-96-val-0.27274127925435704.pt'))

    # model.egenerate(source='live', outputDir=outputDir, outputFilename=f'_pred', resize=(224, 224), show=True, colormap='grayscale')

    # model.egenerate(source='video', inputFilePath='/home/ehsanasgharzde/Desktop/Projects/edepth/input/18839247834792849281913.mp4', outputDir=outputDir, outputFilename=f'18839247834792849281913_pred', resize=(224, 224), show=True, colormap='grayscale')

    totalSec, totalImg = 0, 0
    for index, imagePath in enumerate(inputs):
        if imagePath.replace("\\", "/").split("/")[-1].split(".")[-1] != 'jpg':
            continue
        imageName = imagePath.replace("\\", "/").split("/")[-1].split(".")[0]
        print(index, "- Processing", imageName, end=" - ")
        startTime = time.monotonic()
        model.egenerate(source='image', inputFilePath=imagePath, outputDir=outputDir, outputFilename=f'{imageName}_pred', resize=(224, 224), save=True, colormap='grayscale')
        endTime = time.monotonic()
        totalSec += endTime - startTime 
        totalImg += 1
        print("Process finished in", endTime - startTime, "seconds.")

    print(f"\nGenerated total {totalImg} images in {totalSec} seconds.")
