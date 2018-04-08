

## 1.3.2 神经网络表示  

`$\space$`**浅层神经网络的表示如下**： 

![image](https://thumbnail0.baidupcs.com/thumbnail/69bd87e798999c369242534e04620c72?fid=2218538830-250528-937969313690211&time=1522735200&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-r%2F0UZ3UE9rdm9S6Dxgowg3dzgJQ%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9064973981034741715&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)

#### 需要理清参数的维度关系以及怎样输入：
- [x] 输入层与隐含层之间的连接参数 
    - `$W^{[1]} : (n_{out},n_{in})$`
    - `$b^{[1]}:(n_{out},1)$`，和隐藏层的神经元个数相同

- [x] 逻辑回归中 VS 浅层神经网络的参数 
    - 逻辑回归 `$W:(n_x,1)$` ，`$Z=W^TX+b$`
    - 浅层中，隐含层中涉及到多个神经元，`$W^{[1]}$`参数每列是已经转置的参数 `$Z^{[1]}=W^{[1]}X+b^{[1]}$`
    

#### 数学计算
`$\space$`对单个样本`$x^{(i)}$`:


```math

z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1] (i)}

a^{[1] (i)} = \tanh(z^{[1] (i)})

z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2] (i)}

\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})

y^{(i)}_{prediction} = \begin{cases} 1 & \space {if } a^{[2](i)} > 0.5 \\ 0 & \space{otherwise } \end{cases}

```

`$\space$` 给出所有样本的预测值后，计算损失函数`$J$`：

```math
J = - \frac{1}{m} \sum\limits_{i = 0}^{m}(y^{(i)}\log(a^{[2] (i)}) + (1-y^{(i)})\log (1- a^{[2] (i)}) 

```





## 1.3.4 多个栗子中的向量化 

`$\quad $`为了避免循环，对批量样本进行向量化：

![image](https://thumbnail0.baidupcs.com/thumbnail/da230af2e5c1cc889a7876a093ac23c6?fid=2218538830-250528-1010462353998558&time=1522735200&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-UgKe19QPgqGwEwL%2BYwBdMxtOgL8%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9065646454191682046&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)

- [ ]  从循环操作可以看出，对m个样本都是执行相同的过程，所以可以采用向量化方式来进行处理
- [ ] 样本是以每一列的方式来输入 
- 

## 1.3.6 激活函数 

- [ ] 常用的激活函数：
    - [x] `$sigmoid$`
    - [x] `$tanh: a= \frac{e^z-e^{-z}}{e^z+e^{-z}}$`
    - [x] `$ Relu:a=max(0,Z) $`
    - [x] `$ Leaky\space Relu: a=max(0.01Z,Z)$`
    
![image](https://thumbnail0.baidupcs.com/thumbnail/d0e644aab0e51ae7d9f0bf2bfceaa4b5?fid=2218538830-250528-58217531707296&time=1522738800&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-jBJKVHYPjQYQvhziVoj68mSl26o%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9066045176417177029&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


- [ ] `$\space$`为什么要使用激活函数 ？
    - [x]  没有激活函数整个神经网络就是线性函数的叠加，最后还是线性函数 
    

## 1.3.6 激活函数的导数

- [x] `$sigmoid:a=g(z) , g'(z)=a(1-a)$`  
- [x] `$tanh: g'(z)=1-a^2$`
- [x] `$ Relu:a=max(0,Z) $`
    - `$g'(x)=\begin{cases}1\space ,x> 0 \\ 0\space ,x\le0 \end{cases} $`
- [x] `$ Leaky\space Relu: a=max(0.01Z,Z)$`


`$\quad$` **激活函数的选择：**
- [x] sigmoid 和 tanh 函数当 `$|Z|$` 很大的时候，梯度会很小，在进行梯度更新的时候，后期会变的很慢
- [x] ReLu 弥补了前两者的缺陷，当`$Z>0$`时梯度始终为1  ，但是缺陷是当`$Z<0$`时候 梯度一直为0，不过实际应用中，影响不大 




## 1.3.10 直观理解反向传播

#### 神经网络反向梯度下降公式及其代码向量化

![image](https://thumbnail0.baidupcs.com/thumbnail/b2f020c7798078aa13692932a95e52d0?fid=2218538830-250528-232707305172288&time=1522742400&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-tGeSOX63c%2BQVYGbyoaZLrQK8noE%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=9067530300025628270&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


## 1.3.11 随机初始化：

- [x] 如果不随机初始化参数的全部置为0，两个神经元的影响是相同的，通过反向传播计算梯度的时候，参数更新后任然是对称的，那么设置的多个神经元没有意义


```python 
W1 = np.random.randn(n_h,n_x)*0.01 
b1 = np.zeros((n_h,1))

W2=np.random.randn(n_y,n_h)*0.01 
b2=np.zeros((n_h,1))


```

- 初始化 在 0.01 范围内是因为使用正切 和 sigmoid激活是最好让值落在0附近，梯度更新的快一些 




#### date:2018/04/03 