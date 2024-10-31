from keras.models import Sequential
from keras.layers import Dense
import numpy

import shap
import matplotlib.pyplot as plt

#初始化随机数生成器

#随机梯度下降的随机种子
seed = 7
numpy.random.seed(seed)

#加载数据
dataset = numpy.loadtxt("FeaturesWithLabels-1.csv", delimiter=",")


#将数据分为输入和标签
X = dataset[:,1:]
Y = dataset[:,0:1]


#创建模型
model = Sequential()
model.add(Dense(222, activation='relu')) 
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#训练模型
model.compile(optimizer='RMSProp',loss='binary_crossentropy',metrics=['accuracy']) # Fit the model
model.fit(X, Y, epochs=1000, batch_size = 10, verbose=1)

#评估模型
scores = model.evaluate(X, Y, batch_size=3)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# 创建背景数据集
background = X

# 创建SHAP解释器
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="bar")

'''
feature_names = Y.columns
          
# 手动创建一个 shap.Explanation 对象，并传递特征名
          
shap_values_Explanation = shap.Explanation(values=shap_values_raw.values, base_values=shap_values_raw.base_values,data=X_explain_np,feature_names=feature_names) 

plt.figure(figsize=(10, 5), dpi=1200)
          
# 使用 shap_values_Explanation 绘制摘要图，显示每个特征的影响力
          
shap.summary_plot(shap_values_Explanation, X_test[:100], feature_names=feature_names, plot_type="dot", show=False)
          
plt.savefig("SHAP_Summary_Plot.pdf", format='pdf', bbox_inches='tight')
          
plt.show() # 特征名称'''