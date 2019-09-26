This project is dnn，which is a convient model. You can choose some parameters for your owen application.
  layer_dims: denotes number of each layers.
  classes: denotes number of classes
  beta: denotes the l2 regularize parameter
 
 
 这是一个通用的DNN模型，你只需要做简单修改就能将整个代码跑通。
 其中做一下参数说明：
    classes: 是你要进行几分类的数目
    dirPath： 是你的数据所在的目录路径
    featureName: 是你的训练数据的文件名字，要求其key与文件名相同，否则会出错
    labelName：是你的标签数据的文件名字，同样也要求key与文件名相同
    
    layer_dims： 是神经网络的参数，layer_dims[x_train.shape[0], 10, 3, classess]，表示构建了一个三层的神经网络，第一层隐含层包含10个神经元...
    
    beta：是进行l2正则化的参数，当其取值为0时， 则不进行l2正则化
