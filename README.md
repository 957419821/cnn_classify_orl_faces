# cnn_classify_orl_faces
<p>数据集使用的是 https://cs.nyu.edu/~roweis/data.html 的Olivetti Faces
<p>压缩包可以在http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html 上面下载

p>dataSetTools.tools包含的函数，使用时仅需要调用getData即可，它返回的值经过了平移和加噪处理：</p>
* getLabels(filedir) 被getData、getFeatures、getSize三个函数调用
* getSize(filedir) 被getFeatures调用
* translation(image, vector) 被augment调用
* addNoises(features) 同上
* augment(features, translationTimes=9) 被getData调用
* getFeatures(filedir) 同上
* getData(filedir) 被文件夹外的get_and_restore_data_into_csv.py模块调用
<p>get_and_restore_data_into_csv.py 调用上面提到的模块，并将得到的经过提升之后的数据存入csv文件
<p>network.py中是简单的CNN，它将调用dataset.py模块，并使用其类得到数据</p>
dataset.py模块中包含：
* 函数getCSVdata 用于从csv文件中得到数据，它被类DataSet调用
* 类DataSet 存储数据的信息，被network.py使用
