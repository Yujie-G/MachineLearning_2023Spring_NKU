# Lagrange Multiplier Method
## 原题
让直线$ x+y=1$与椭圆$x^2 + 2y^2=C (C)$相切，求$C$的取值。采用拉格朗日乘子法求解。

## Solution

转为带参数的优化问题：

$$
min\quad x^2 + 2y^2  
\\\  
s.t. \quad x+y=1  
$$

使用拉格朗日乘数：

$$
L = x^2 + 2y^2 + \lambda(x+y-1)
$$

目标即为最小化$L$,对其求偏导

$$
\frac{ \partial L }{ \partial x } = 2x+\lambda = 0  
\\\\\  
\\
\frac{ \partial L }{ \partial y } = 4y+\lambda = 0  
$$

因此有
$$
\begin{cases}
2x=4y \\\  
x+y=1
\end{cases}
$$

解得
$$
\begin{cases}
x=\frac{2}{3} \\\  
y=\frac{1}{3}
\end{cases}
$$

代入椭圆方程得：

$$ C= \frac{2}{3}$$