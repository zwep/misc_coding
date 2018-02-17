import pykalman 
import numpy as np
import pyplot
import scipy.stats as ss

# Creating a sinus as dummy data
x = np.arange(0,2*np.pi,0.1)
y = np.sin(x)
ydiff = np.insert(np.diff(y),0,values = 0)
ydiff2 = np.insert(np.diff(ydiff),0,values = 0)

# Creating some features...
f0 = np.ones(x.shape[0])
f1 = np.array(1*(y>=0))
f2 = np.array(1*(ydiff>=0))
f3 = np.array(1*(ydiff2>=0))
f4 = f1 * f2
f5 = np.array(1*(np.abs(y)>=0.5))	

X = np.column_stack([f0,f1,f2,f3,f4,f5])

# Making the observation matrix
obs_mat = np.zeros((x.shape[0],1,5+1))+1
obs_mat[:,0,:] = np.copy(X[:,:])

# Defining the model object
kf = pykalman.KalmanFilter(n_dim_state = 6, n_dim_obs = 1,observation_matrices = obs_mat)
# Initializing the parameters
kf = kf.em(y[0:10])
# Get the values for the states... smoothed
smooth_state_mean, smooth_state_cov = kf.smooth(y[0:10])


# This is how it should NOT be
# Now we are just lagging everything..
# And above all.. using future data
y_pred = np.array([])
for i in range(1,x.shape[0]):
	smooth_state_mean, smooth_state_cov = kf.smooth(y[0:i+1])
	y_pred = np.append(y_pred,np.dot(smooth_state_mean[-1],obs_mat[i,0,:]))

plt.plot(y_pred)
plt.plot(y,'r-')
plt.axis([0,10,-1,1])
plt.show()


# This is already better
# Now we are not using future data...
y_pred = np.array([])
y_conf_up = np.array([])
y_conf_down = np.array([])

for i in range(1,x.shape[0]):
	smooth_state_mean, smooth_state_cov = kf.smooth(y[0:i])
	y_pred = np.append(y_pred,np.dot(smooth_state_mean[-1],obs_mat[i,0,:]))
	y_conf = ss.norm.ppf(0.95,y_pred[-1],kf.observation_covariance)
	y_conf_up   = np.append(y_conf_up,y_pred[-1]+np.abs(y_conf[0]))
	y_conf_down = np.append(y_conf_down,y_pred[-1]-np.abs(y_conf[0]))
	 

plt.plot(y_conf_down,'b--')
plt.plot(y_conf_up,'g--')
plt.plot(y_pred,'r-')
#plt.axis([0,10,-1,1])
plt.show()


# This is a step where we incorporate the random
# effects as well..

y_pred = np.array([])
z_score = 1.95
for i in range(1,x.shape[0]):
	smooth_state_mean, smooth_state_cov = kf.smooth(y[0:i])
	up_state = np.random.multivariate_normal(np.zeros(6),kf.transition_covariance,size = 1)
	up_smooth_state_mean = smooth_state_mean[-1] + up_state
	up_obs = np.random.normal(0,kf.observation_covariance,size = 1)
	y_pred = np.append(y_pred,np.dot(up_smooth_state_mean,obs_mat[i,0,:])) + up_obs
	

plt.plot(y_pred)
plt.plot(y,'r-')
plt.show()

plt.plot(y_pred)
plt.plot(y,'r-')
plt.axis([0,10,-1,1])
plt.show()


x = np.linspace(ss.norm.ppf(0.01),ss.norm.ppf(0.99), 100)
ax.plot(x, ss.norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')