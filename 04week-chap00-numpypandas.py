# Numarray와 Numeric이라는 오래된 Python 패키지를 계승해서 나온 
# 수학 및 과학 연산을 위한 파이썬 패키지이다. 
# Py는 파이썬을 나타내기 때문에, 일반적으로 넘파이라고 읽는다.
#
# 프로그래밍 하기 어려운 C,C++,Fortran 등의 언어에 비하면, 
# NumPy로는 꽤나 편리하게 수치해석을 실행할 수 있다. 
# 게다가 Numpy 내부는 상당부분 C나 포트란으로 작성되어 실행 속도도 꽤 빠른편이다. 
# 기본적으로 array라는 자료를 생성하고 이를 바탕으로 
# 색인, 처리, 연산 등을 하는 기능을 수행한다. 
# 물론 넘파이 자체만으로도 난수생성, 푸리에변환, 행렬연산, 간단한 
# 기술통계 분석 정도는 가능하지만 실제로는 Scipy, Pandas, matplotlib 등 
# 다른 Python 패키지와 함께 쓰이는 경우가 많다.

# 하도 기본적으로 쓰이는 모듈이다 보니 Numpy를 보통 np로 호출하는 것이 관례가 되었다. 
# Ex) import numpy as np.


#예제 # numpy 예제

# 배열 생성
import numpy as np
x = np.array([1, 2, 3])
x
y = np.arange(10) # like Python's range, but returns an array
y

# 기본 작업
a = np.array([1, 2, 3, 6])
b = np.linspace(0, 2, 4) # create an array with four equally spaced points starting with 0 and ending with 2.
c = a - b
c
a**2

# 유니버셜 함수
a = np.linspace(-np.pi, np.pi, 100)
b = np.sin(a)
c = np.cos(a)

# 선형 대수학
from numpy.random import rand
from numpy.linalg import solve, inv
a = np.array([[1, 2, 3], [3, 4, 6.7], [5, 9.0, 5]])
a.transpose()
inv(a)
b = np.array([3, 2, 1])
solve(a, b) # solve the equation ax = b
c = rand(3, 3) * 20 # create a 3x3 random matrix of values within [0,1] scaled by 20
c
np.dot(a, c) # matrix multiplication
a @ c # Starting with Python 3.5 and NumPy 1.10

# OpenCV와의 통합
import numpy as np
import cv2 
# ModuleNotFoundError: No module named 'cv2': 설치 필요...
# => (anaconda prompt 관리자 권한으로 ) conda install -c menpo opencv
r = np.reshape(np.arange(256*256)%256,(256,256)) # 256x256 pixel array with a horizontal gradient from 0 to 255 for the red color channel
g = np.zeros_like(r) # array of same size and type as r but filled with 0s for .t-…the green color channel
b = r.T # transposed r will give a vertical gradient for the blue color channel
cv2.imwrite('gradients.png', np.dstack([b,g,r])) # OpenCV images are interpreted as BGR, the depth-stacked array will be written to an 8bit RGB PNG-file called 'gradients.png'
import os
os.getcwd() # 해당 폴더에 gradients.png 파일 있음


# 판다스(pandas) : 파이썬 언어로 작성된 데이터를 분석 및 조작하기 위한 소프트웨어 라이브러리이다. 
# 판다스는 수치형 테이블과 시계열 데이터를 조작하고 운영하기 위한 데이터를 제공하는데, 
# 이름은 계량 경제학에서 사용되는 용어인 'PANel DAta'의 앞 글자를 따서 지어졌다.
# 판다스는 R에서 사용되던 data.frame 구조를 본뜬 DataFrame이라는 구조를 사용하기 때문에, 
# R의 data.frame에서 사용하던 기능 상당수를 무리없이 사용할 수 있도록 만들었다. 
# 판다스의 주요 특성
# 판다스 라이브러리의 주요 코드는 Cython이나 C로 작성되었으며, 퍼포먼스에 최적화되어있다.
# 판다스의 개발자인 웨스 메키니(Wes McKinney)는 금융 데이터에 대한 
# 계량적 분석을 수행하기 위한 고성능의 유연한 툴을 만들 필요가 있다 생각하여, 
# AQR Capital Management에서 근무하던 2008년부터 판다스 개발 작업을 시작하였다. 
# https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

# pandas 예제
import numpy as np       
import pandas as pd       
s = pd.Series([1, 3, 5, np.nan, 6, 8])   
s          
dates = pd.date_range('20130101', periods=6)       
dates          
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))     
df          
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
df2          
df2.dtypes          
#df2.<TAB>  # noqa: E225, E999   
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))       
df
df.head()          
df.tail(3)          
df.index          
df.columns          
df.to_numpy()          
df.to_numpy()          
df.describe()          
df.T      
df.sort_index(axis=1)    
df.sort_index(axis=1, ascending=False)         
df.sort_values(by='B')          
df['A']          
df[0:3]          
df['20130102':'20130104']          
df.loc[dates[0]]          
df.loc[:, ['A', 'B']]        
df.loc['20130102':'20130104', ['A', 'B']]        
df.loc['20130102', ['A', 'B']]        
df.loc[dates[0], 'A']         
df.iloc[3]          
df.iloc[3:5, 0:2]         
df.iloc[[1, 2, 4], [0, 2]]      
df.iloc[1:3, :]         
df.iloc[:, 1:3]         
df.iloc[1, 1]         
df.iat[1, 1]
