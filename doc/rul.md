<div align="justify">

## Why RUL estimation?

## Mathematical formulation for RUL estimation problem
Suppose that we are provied with run-to-failure historical data of multiple machines of the same type. The historical data is supposed to cover a representative set of the considering machine type. The recored data of each machine ends when it reach a failure condition threshold. The objective is to learn a RUL model to predict the remaining lifetime of the considering machine type. 

### Training data
The training data is a set of the operation history of $N$ machines of the same type denoted as $\mathcal{X} = \left\lbrace X^n \mid n=1, ..., N \right\rbrace$. There are $M$ measurements that quantify their health behaviour during their operations (e.g. sensors that are installed on these machine to monitor their conditions). The data from the $n^{th}$ machine throughout its usefull lifetime produces a multivariate timeseries $X^n \in \mathbb{R}^{T^n \times M}$ in which $T^n$ denotes the total number of time steps of machine $n$ throughout is lifetime (in other words, $T^n$ is the failure time of component $n$). We use the notation $X_t^n \in \mathbb{R}^M$ to denote the $t^{th}$ timestep of $X^n$ where $t \in \left\lbrace 1, ..., T^n \right\rbrace$. Indeed, $X_t^n$ is a vector of $M$ sensor values.

### Testing data
The test set consists of historical data of $K$ machines of the same considering type denoted as $\mathcal{Z}= \left\lbrace Z^k \mid n=1, ..., K \right\rbrace$ in which $Z^k$ is a time series of $k^{th}$ machine. The notation $Z_t^k \in \mathbb{R}^M$ is used to denote the $k^{th}$ timestep of $Z^k$ where $t \in \left\lbrace 1, ..., L^k \right\rbrace$ where $L^k$ is the total number of time steps related to machine $k$ in the test set. Obviously, the test set will not consist of all time steps upto the failure point, i.e.., $L^k$ is generally smaller than the the failure time of component $k$ denoted as $\bar{L}^k$.

We focus on estimating RUL of component $k$, $\bar{L}^k - L^k$, given the data from time step 1 to $L^k$. Note that $\bar{L}^k - L^k$ is also provided in the test set.

## Data preprocessing
### RUL target
We can generate the RUL for very time steps in a training trajectory $X^n$ based on $T^n$. In the literature, there are two common models for generating RUL given the failure time, namely, **linear** and **piecewise linear** model degradation model. These two models are mathematically presented in the following.

1. Linear model degradation model  
   This kind of RUL model is very obvious considering the fact that we have the the failure point of each training trajectory $(T^n)$. The RUL of machine $n$ at time step $t$ in the training set, $R_t^n$, is calculated as belows:

   $$
   R_t^n = T^n - t
   $$

2. Piecewise linear degradation model  
   Since the degradation of a machine will generally not be noticeable unit it has been operating for some period of time. Therefore, it is probaly reasonable to estimate RUL of a machine unitl it begins to degrade. For this reason, it seem to be ok to estimate the RLU when the machine is new as constant. As a result, the piecewise linear degradation model is propsoed to set an upper limit on the RUL target as belows:  

   $$
   R_t^n = 
   \begin{cases}
   R_t ^n = R^{max}  & \text{if } t \le T^{max} \\ 
   R_t^n = T^n - t   & \text{otherwise} 
   \end{cases}
   $$

   in which $T^{max}$ is the maximal time step for considering the machine as new and $R^{max}$ is the upper litmit of RUL.
### Data normalization
### Data augmentation