# //导入要用到的库
from sklearn.linear_model import LinearRegression
import pandas as pd
# 用pandas库把Data.csv文件里的内容读出来
full_data = pd.read_csv("Data.csv")
model = LinearRegression()
# 创建一个线性回归的模型，调用它的fit方法训练它，注意我们要用性别和身高预测体重，
# 所以fit方法的第一个参数是性别和身高，第二个参数是体重
model.fit(full_data[['Gender', 'Height']], full_data["Weight"])
print("编译成功拉~~~~")
# 预测身高172cm男性的体重
# 推测体重是66.811415kg。
# 根据生成数据的逻辑，172cm的男性的体重应该是67kg，与预测的情况基本一致。
# [67.10708506]
print(model.predict([[1, 172]]))

import coremltools
coreml_model = coremltools.converters.sklearn.convert(model, ["gender", "height"], "weight")

coreml_model.author = "test"
coreml_model.license = "BSD"
coreml_model.short_description = 'Predicts weight by gender and height'

coreml_model.input_description['gender'] = "Gender: 0 for male; 1 for female."
coreml_model.input_description['height'] = 'Height in cm'
coreml_model.output_description['weight'] = 'weight in kg'
coreml_model.save('PredictWeight.mlmodel')

编译成功拉~~~~
[67.10708506]
WARNING:root:scikit-learn version 0.23.1 is not supported. Minimum required version: 0.17. Maximum required version: 0.19.2. Disabling scikit-learn conversion API.
Traceback (most recent call last):
  File "coreML.py", line 18, in <module>
    coreml_model = coremltools.converters.sklearn.convert(model, ["gender", "height"], "weight")
  File "/Users/sfyh/anaconda3/lib/python3.8/site-packages/coremltools/converters/sklearn/_converter.py", line 148, in convert
    from ._converter_internal import _convert_sklearn_model
  File "/Users/sfyh/anaconda3/lib/python3.8/site-packages/coremltools/converters/sklearn/_converter_internal.py", line 36, in <module>
    from . import _decision_tree_classifier
  File "/Users/sfyh/anaconda3/lib/python3.8/site-packages/coremltools/converters/sklearn/_decision_tree_classifier.py", line 16, in <module>
    sklearn_class = _tree.DecisionTreeClassifier
NameError: name '_tree' is not defined
(base) lengwujudeMacBook-Pro:训练模型 sfyh$ pip install joblib
Requirement already satisfied: joblib in /Users/sfyh/anaconda3/lib/python3.8/site-packages (0.16.0)
(base) lengwujudeMacBook-Pro:训练模型 sfyh$ 
