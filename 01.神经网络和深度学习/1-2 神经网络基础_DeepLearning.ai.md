

## 1.2.1 二分分类
`$\quad$` 在二分类问题中，输出是一个非连续的值，课件中给出了一个`$Notation$`：
- [x] 给定`$m$`个样本`$(x,y)$`，其中`$x\in \mathbb R^{n_x}$`，表示样本包含`$n_x$`个特征
- [x] `$y \in0,1$`，目标值属于0,1分类
- [x] `$X.shape=(n_x ,m)$`


![image](https://thumbnail0.baidupcs.com/thumbnail/9bf3c3de70929d69b4b112340d8f8e69?fid=2218538830-250528-930613641238610&time=1522720800&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-K5WFXI7gom0c2XbKUw1vgnPEwP8%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9061403552129359584&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)

- 在图片分类中 `$cat\space vs \space Noncat$`中，将图片装换为 红黄蓝像素矩阵拼接，特征维度为： `$n_x =64\times 64\times 3=12288$`


## 1.2.2 Logistic 回归 

`$\space$` **逻辑回归中**:
- [ ] 给定`$x$`,想得出`$\hat y=p(y=1,x)$`,其中`$x\in \mathbb R^{n_x}$`
- [ ]  输出： `$\hat y=\sigma(W^T x+b)$`,其中 `$\delta(z)=\frac1{1+e^{-z}}$`

- 其中 `$sigmoid$`的一阶导数可以用自身来表示：`$\sigma'(z)=\sigma(z)(1-\sigma(z))$` ，由于导数 `$max(\sigma'(z))=0.25$` ，随着梯度下降公式的不断跟新，梯度容易消失 `$\space$` (`$?$`所以神经网络中间层中一般不用`$sigmoid$`激活函数)


## 1.2.3 Logistic 损失函数

`$\quad $`为了训练得到参数，我们需要定义损失函数：
给定{`$ (x^{(1)},y^{(1)}),...(x^{(m)},y^{(m)})$`},我们希望`$\hat y^{(i)}\approx y^{(i)}$`

### Loss(error) function: 

`$\space$`损失函数衡量了预测输出与实际输出之间的误差，对与单个样本来说：
- [x] 平方误差： `$\mathbf L(y^{(i)},\hat y^{(i)})=(y^{(i)}-\hat y^{(i)})^2$`
- [x] Logistic 损失函数 `$\mathbf L(y^{(i)},\hat y^{(i)})=-(y^{(i)}log\hat y^{(i)})+(1-y^{(i)}log(1-\hat y^{(i)}))$`

- 对于逻辑回归来说，平方误差损失函数是==非凸==函数，使用梯度下降的时候容易得到局部最优解


### Cost function:
`$\quad $`代价函数是在整个训练样本上的平均损失函数,通过优化损失函数，找到参数`$w,b$`,使得损失函数最小化：


`$\qquad J(w,b)=-\frac 1{m}\sum_{i=0}^m [(y^{(i)}log\hat y^{(i)})+(1-y^{(i)}log(1-\hat y^{(i)}))]$` 



## 1.2.4 梯度下降法

`$\space$`通过梯度下降更新参数：
-  `$w:= w- \alpha\frac{dJ(w,b)}{dw}$`
-  `$b:= b-\alpha\frac{dJ(w,b)}{db}$`



## 1.2.9 Logistic 回归中的梯度下降法 

`$\space $`**对于单个样本,损失函数：**

`$L(a,y)=-(ylog(a)+(1-y)log(1-a))$`，其中`$a=\sigma(z)$` ， 对`$da\space  dz $`求导：

 ![image](https://thumbnail0.baidupcs.com/thumbnail/b484122c1802a24e4cd598f373720869?fid=2218538830-250528-1029129659348562&time=1522720800&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-KBz3JsfOyOrJuarbGA62zh%2BFVR0%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9061430557877639522&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)

- [x] `$da=\frac{\partial L}{\partial a}=-\frac{y}{a}+\frac{1-y}{1-a}$`
- [x] `$dz=\frac{\partial L}{\partial z}=a-y$`

`$\quad$`再对`$w_1$`、`$w_2$`求导：
- [x] `$dw_1=x_1dz=x_1(a-y)$`
- [x] `$dw_2=x_2dz=x_2(a-y)$`
- [x] `$db=dz=(a-y)$`

`$\quad$`更新参数：

```math
w_1:=w_1-\alpha dw_1

w_2:=w_2-\alpha dw_2

b:=b-\alpha db
```

`$\space $`**对于m个样本,跟新参数取平均：**

- [x] `$dw_1=\frac1{m}\sum_{i=1}^m x_1^{(i)}(a^{(i)}-y^{(i)})$`
- [x] `$db=\frac1{m}\sum_{i=1}^m(a^{(i)}-y^{(i)})$`


## 1.2.10 向量化Logistic 

- 输入矩阵`$X:(n_x,m)$`
- 权重矩阵`$W:(n_x,1)$`
- 偏置 b ：为常数
- 输出矩阵`$Y:(1,m)$`  
- 

`$\space $`**逻辑回归梯度下降输出向量化：**

- [x] `$dz:(1,m)$ `   ，`$dz=A-Y$`
- [x] `$db=\frac1{m}\sum_{i=1}^mdz^{(i)}$`
- [x] `$dW =\frac1{m}X.dZ^T$` 


### python 的一些细节处理

- 为了及时检查矩阵的维度使用 assert  reshape 等  

```python 
assert(a.shape==(5,1))
a.reshape(5,1)
```


## 1.2.11 Logistic  代价函数的解释 

`$\space$` 给定`$x$` , `$y$`的概率，其中我们用`$\hat y$`看作输出正类的概率：

```math
P(y|x) = \hat y^y(1-\hat y)^{(1-y)}  ,
```
`$P(y|x)$`概率越大越好，损失函数我们可以取 `$log$` 后加负号定义为损失函数，越小越好：

```math
L(\hat y,y)=-(ylog\hat y+(1-y)log(1-\hat y))\space ,
```
假设`$m$`个样本是独立同分布的，则有：

```math
max \prod_{i=1}^{m} P(y^{(i)},y^{((i)}) \space ,
```
对上式去负和 `$log$` 后，得：


```math
J(w,b)=-1/m\sum_{i=1}^{m}[y^{(i)}log\hat y^{(i)}+(1-y^{(i)})log(1-\hat y^{(i)})]
```



**Done: 2018/04/02**
