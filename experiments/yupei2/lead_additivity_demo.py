import numpy as np
T = 600
tau = 1.0            # 秒
db = 0.01            # 10ms 桶 -> 秒
d = np.exp(-db/tau)

def buckets(intervals):
    u = np.zeros(T)
    for a,b in intervals:
        u[a:b] = 1.0
    return u

A = buckets([(0,100),(300,400)])       # A 活跃两段
B = buckets([(1,101)])                 # B 只跟第一段
C = buckets([(301,401)])               # C 只跟第二段

def Gsum(u):
    g = np.zeros(T); s = 0.0
    for k in range(T):
        g[k] = s; s = s*d + u[k]
    return g

def Gmax(u):
    g = np.zeros(T); m = 0.0
    for k in range(T):
        g[k] = m; m = max(u[k], m*d)
    return g

def S_sum(ua, ub):                      # 求和核（含同桶折半）
    return float(ua @ Gsum(ub) + 0.5*(ua*ub).sum())

def cov(ua, gb):                        # 取max覆盖：每笔A找最近的一笔池内成交
    return float(ua @ gb)

UB = np.clip(B+C, 0, 1)
print("求和核:  S(A,B)=%.4f  S(A,C)=%.4f  S(A,B)+S(A,C)=%.4f  S(A,UBC)=%.4f" %
      (S_sum(A,B), S_sum(A,C), S_sum(A,B)+S_sum(A,C), S_sum(A,UB)))
print("覆盖核:  cov(A,B)=%.4f cov(A,C)=%.4f max=%.4f  cov(A,UBC)=%.4f" %
      (cov(A,Gmax(B)), cov(A,Gmax(C)), max(cov(A,Gmax(B)),cov(A,Gmax(C))),
       cov(A, np.maximum(Gmax(B),Gmax(C)))))
print("A 的总权重 = %.1f" % A.sum())
