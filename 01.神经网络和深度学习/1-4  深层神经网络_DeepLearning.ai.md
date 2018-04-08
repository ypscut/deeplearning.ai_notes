
### 整体的流程图

![image](https://thumbnail0.baidupcs.com/thumbnail/4c267f329c42fac366ccbc5b7881bce0?fid=2218538830-250528-75855232729853&time=1523005200&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-qGPtGXmr%2B6fPydJGdO3nPhuDcSE%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9137637511868553915&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


### 1.4.4 为什么使用深层表示 

`$\quad$`在人脸识别中，神经网络的第一层从原始图片中提取一些边缘轮廓，可以把它理解为边缘探测器，后面的深层能将第一层学习到的边缘轮廓组合起来形成一些脸部的局部特征，例如眼睛，耳朵，可以理解为面部探测器，从简单到复杂，从局部到整体：

![image](https://thumbnail0.baidupcs.com/thumbnail/34fa571d717c030085ff21485a2401da?fid=2218538830-250528-880107609573433&time=1523005200&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-kXfttSt9ACBqC4tudZikMDF75eg%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9137990418966203936&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)

`$\quad$`对于语音识别中，第一层可以学习到语音发音中的音调，更深层中网络能检测到基本的单词信息再逐步学习到短语句子。



### 1.4.5 搭建深层结构网络块  

`$\quad$`神经网络中的参数表示
- `$n^{[l]}$`表示的是第`$l$`层神经元个数
- `$a^{[l]}$`表示的是第`$l$`层激活函数的输出 
- `$W^{[l]}$`表示的是第`$l$`层的权重 
- `$b^{[l]}$`表示的是第`$l$`层的偏置
- 输入`$x$`记为`$a^{[0]}$`，输出`$\hat y$`记为`$a^{[L]}$`


- [x] **前向传播（Forward propagation）**
    - 输入`$a^{[l-1]}$`
    - 输出 `$a^{[l]}$`,cache(`$z^{[l]}$`) 
    

`$\qquad$`公式：
```math
z^{[l]} = W^{[l]}.a^{[l-1]}+b^{[l]}

a^{[l]} = g^{[l]}(z^{[l]})
```

`$\qquad$`向量化：


```math
Z{[l]} = W^{[l]}.A^{[l-1]}+b^{[l]}

A^{[l]} = g^{[l]}(Z^{[l]})
```

![image](https://thumbnail0.baidupcs.com/thumbnail/024a350a90ce97342147843af110effd?fid=2218538830-250528-383211816413233&time=1523073600&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-OGpsh0XuO6EpxQJbdDCyZccyU0c%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9156363125202521459&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


- [x] **反向传播（Backward propagation）**
    - 输入`$da^{[l]}$`
    - 输出 `$da^{[l-1]}$`,`$dW^{[l]}$`,`$db^{[l]}$`
    

`$\qquad$`公式：
```math
dz^{[l]} = da^{[l]}*g^{[l]'} (z^{[l]})

dW^{[l]} = dz^{[l]}.a^{[l-1]}

db^{[l]} = dz^{[l]}

da^{[l-1]} = W^{[l]T}.dz^{[l]}

```

`$\qquad$`向量化：


```math
dZ^{[l]} = dA^{[l]}*g^{[l]'} (Z^{[l]})

dW^{[l]} = dZ^{[l]}.A^{[l-1]}

db^{[l]} =1/m.np.sum( dZ^{[l]},axis=1,keepdims=True)

dA^{[l-1]} = W^{[l]T}.dZ^{[l]}


```

    


### 1.4.7 参数 VS 超参数

`$\quad$` **超参数：**
- 学习率 `$\alpha$`
- 迭代次数 `$N$`
- 网络层数 `$L$`
- 隐藏层中神经元个数 `$n^{[1]}\space n^{[2]}\cdots$`
- 激活函数的选择 `$g(z)$`

- [x] 这些超参数最终决定着 `$W\space b$`





## 代码 ：

![image](https://thumbnail0.baidupcs.com/thumbnail/092ccbcecc377c2ebf704d553ae4c508?fid=2218538830-250528-196101696835328&time=1523070000&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-aPVsl5LWvLX3lS%2FZFDF%2BnDNBwe0%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9155397278710869801&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


```
# 将每张图片  64*64*3 转换成 12288*1  

Each image is of size: (64, 64, 3)

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions

```

