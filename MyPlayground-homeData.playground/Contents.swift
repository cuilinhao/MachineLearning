import Cocoa
import CreateML
import CreateMLUI
import Foundation


//1
/**
 1. 首先，我们创建一个名为data的常量，它是垃圾邮件的一种MLDataTable。json文件。MLDataTable是一个全新的对象，用于创建一个决定训练或评估ML模型的表。我们将数据分为trainingData和testingData。和以前一样，比率是80-20，种子是5。种子是指分类器的起点。然后我们用我们的训练数据定义一个叫做spamClassifier的MLTextClassifier，定义数据的值是文本，什么值是标签。
 */
let houseData = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/sfyh/Desktop/训练模型/HomeDataTraining/HouseData.csv"))

let (trainingCSVData, testCSVData) = houseData.randomSplit(by: 0.8, seed: 0)

//2
/**
 创建了两个变量，名为trainingAccuracy和validationAccuracy，用于确定分类器的准确程度。在侧窗格中，您可以看到百分比
 */
let pricer = try MLRegressor(trainingData: houseData, targetColumn: "MEDV")

//3
/**
 我们还检查评估的执行情况。请记住，评价是分类器以前没有看到的文本上使用的结果，以及它们的准确性
 */
let csvMetadata = MLModelMetadata(author: "天下林子", shortDescription: "A model used to determine the price of a house based on some features.", version: "1.0")

//try pricer.write(to: URL(fileURLWithPath: "/Users/Path/To/Write/HousePricer.mlmodel"), metadata: csvMetadata)
//4.  保存
try pricer.write(toFile: "/Users/sfyh/Desktop/训练模型", metadata: csvMetadata)



