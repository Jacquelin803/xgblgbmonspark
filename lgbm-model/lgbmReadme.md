## 记录踩坑
### 依赖
#### 主要有三部分： 
##### 1.spark  
##### 2.jpmml：生成pmml模型文件用，需要jpmml-lightgbm以及jpmml-sparkml  
##### 3.lightgbm：lightgbm源码、mmlspark  
##### 需要注意的点：lgbm本身用mmlspark框架，算法二分类多分类回归都可实现，生成pmml的依赖版本不同于xgboost可以使用多个版本，lgbm用的jpmml-sparkml我这边只有1.5.0测试通过（我测试过1.5.[0-13]），版本限制见https://github.com/jpmml/jpmml-sparkml ，所以github上的我用了两个module
### 训练模型  
##### 过程是Scala ML的常规套路，只是转换pmml文件需要注意，找了其他方式（比如https://github.com/alipay/jpmml-sparkml-lightgbm ）都不行，看了下是因为这个包的源码是三年前的了，重要函数缺失，当然如果有其他方式欢迎提出，毕竟现在这种情况，如果同时使用多个算法就得为lgbm的pmml生成另写一套，无法写基类去约束就比较乱，其他算法我用的PMMLBuilder。
