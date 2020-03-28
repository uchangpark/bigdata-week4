
# In[2]:
# unit 21 배열만들기
import numpy as np
numbers = np.array(range(1, 11), copy=True)
print('*number=>', numbers)
numberlist=[x for x in range(1,11)]
print('*numberlist=>', numberlist)

#---------------------------------------
ones = np.ones([2, 4], dtype=np.float64)
print('*ones=>', ones)
zeros = np.zeros([2, 4], dtype=np.float64)
print('*zeros=>', zeros)
# 배열의 내용물이 항상 0인 것은 아니다.
empty = np.empty([2, 4], dtype=np.float64)
print('*empty=>', empty)

#---------------------------------------
print('*ones.shape=>', ones.shape) # 아직 변형되지 않았다면 원래 모양을 반환한다.
print('*number.ndim=>', numbers.ndim) # len(numbers.shape)와 같다.
print('*zeros.dtype=>', zeros.dtype)
eye = np.eye(3, k=0)
print('*eye=>', eye)
np_numbers = np.arange(2, 5, 0.25)
print('*np_numbers=>', np_numbers)
np_inumbers = np_numbers.astype(np.int)
print('*np_inuumbers=>', np_inumbers)

# In[3]:
# unit22 행변환
import numpy as np
sap = np.array(["11", "22", "33", "44", "55", "66", "77", "88"])
print('*sap=>', sap)
sap2d = sap.reshape(2, 4)
print('*sap2d=>',sap2d)
sap2dswap=sap2d.swapaxes(1, 0)
print('*sap2d,swaxes=>',sap2dswap)
sap2dt=sap2d.T
print('*sap2d.transpose=>',sap2dt)
sap3d = sap.reshape(2, 2, 2)
print('*sap3d=>',sap3d)

# In[4]:
# unit 23 indexing, slicing
#---------------------------------------
dirty = np.array([9, 4, 1, -0.01, -0.02, -0.001]); 
print('*dirty=>', dirty)
whos_dirty = dirty < 0 # 불 배열을 불 인덱스로 사용한다.
print('*whos_dirty=>', whos_dirty)
dirty[whos_dirty] = 0 # 모든 음수값을 0으로 바꾼다.
print('*dirty=>', dirty)
linear = np.arange(-1, 1.1, 0.2)
print('*linear=>', (linear <= 0.5) & (linear >= -0.5))

# In[6]:
# unit 24 브로드캐스팅
import numpy as np
a = np.arange(4); 
print(a)
b = np.arange(1, 5) ; 
print(b)
print('*a+b=>', a+b); 
print('*a*5=>', a*5)
c=[1,2,3,4]; # 리스트 타입 곱셈과 비교
print('*c*5=>', c*5)
noise = np.eye(4) + 0.01 * np.ones((4, )); 
print('*noise=>', noise)
noise = np.eye(4) + 0.01 * np.random.random([4, 4]); 
print('*noise=>', noise)
print('*np.round=>', np.round(noise, 2))

# In[7]:
# unit 25 유니버셜함수
stocks = np.array([140.49, 0.97, 40.68, 41.53, 55.7, 57.21, 98.2, 99.19])
print('*stocks=>', stocks)
stocks = stocks.reshape(4, 2).T
print('*stocks=>', stocks)
fall = np.greater(stocks[0], stocks[1])
print('*fall=>', fall)

# 결측치 -> 0으로 변환
stocks[1, 0] = np.nan
print('*np.isnan(stocks)=>', np.isnan(stocks))
stocks[np.isnan(stocks)] = 0
print('*stocks=>',stocks)

# In[8]:
# unit 26 조건부함수
changes = np.where(np.abs(stocks[1] - stocks[0]) > 1.00, stocks[1] - stocks[0], 0)
print('*changes=>', changes)
newsap = np.array(["11", "22", "33", "44"])
print('*newsap=>', newsap[np.nonzero(changes)])
print('*newsap=>', newsap[np.abs(stocks[1] - stocks[0]) > 1.00])

# In[43]:
# unit 30 합선사인파만들기
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 신호, 잡음, 그리고 "악기" 정보를 상수로 정의한다.
SIG_AMPLITUDE = 10; SIG_OFFSET = 2; SIG_PERIOD = 100
NOISE_AMPLITUDE = 3
N_SAMPLES = 5 * SIG_PERIOD
INSTRUMENT_RANGE = 9
# 사인 곡선을 구성하고 잡음을 섞어 넣는다.
times = np.arange(N_SAMPLES).astype(float)
signal = SIG_AMPLITUDE * np.sin(2 * np.pi * times / SIG_PERIOD) + SIG_OFFSET
noise = NOISE_AMPLITUDE * np.random.normal(size=N_SAMPLES)
signal += noise
# # 음역대를 벗어난 스파이크를 제거한다.
signal[signal > INSTRUMENT_RANGE] = INSTRUMENT_RANGE
signal[signal < -INSTRUMENT_RANGE] = -INSTRUMENT_RANGE
# 결과를 플롯(plot)으로 시각화한다.
matplotlib.style.use("ggplot")
plt.plot(times, signal)
plt.title("Synthetic sine wave signal")
plt.xlabel("Time")
plt.ylabel("Signal + noise")
plt.ylim(ymin = -SIG_AMPLITUDE, ymax = SIG_AMPLITUDE)
# 플롯을 저장한다.
plt.savefig("signal.pdf")

# In[ ]:

