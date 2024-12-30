import numpy as np
def gauss_sample(mu, sigma, n):
    a = np.linalg.cholesky(sigma)
    z = np.random.randn(len(mu), n)
    k = np.dot(a, z)
    return np.transpose(mu + k)


def gauss_condition(mu,sigma,hidden_indexes,visible_indexes,visible_values):

     v = visible_indexes.reshape(len(visible_indexes))
     h = hidden_indexes.reshape(len(hidden_indexes))

     if len(h)==0:
         mugivh = np.array([])
         sigivh = np.array([])
     elif len(v) == 0:
         mugivh = mu
         sigivh = sigma
     else:
         #get sigma_11
         ix_hh = np.ix_(h,h)
         sigma_hh = sigma[ix_hh]

         #get sigma_12
         ix_hv = np.ix_(h,v)
         sigma_hv = sigma[ix_hv]

         #get sigma_22
         ix_vv = np.ix_(v,v)
         sigma_vv = sigma[ix_vv]

         sigma_vv_inv = np.linalg.inv(sigma_vv)

         visible_values_len = len(visible_values)
         inner =  (visible_values.reshape((visible_values_len,1))-mu[v].reshape((visible_values_len, 1)))
         outer = np.dot(sigma_hv, np.dot(sigma_vv_inv,inner)).flatten()
         mugivh = mu[h] + outer
         sigivh = sigma_hh - np.dot(sigma_hv, np.dot(sigma_vv_inv, np.transpose(sigma_hv)))
         
     return mugivh, sigivh
    