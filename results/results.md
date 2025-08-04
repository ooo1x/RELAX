# 2025.7.18
## 1. Default Fault Injector 

**Fault injector**：state 4 or 5  
**Paramaters**：

fault_duration_min: 10
fault_duration_max: 10
fault_amplitude_min: 0.3
fault_amplitude_max: 0.3
joint_names:

panda_joint4


## 2. obstacles positions:
    self.obstacle1 = np.array([0.75, -0.25, 1.48])
    self.obstacle2 = np.array([0.75,  0.25, 1.48])
    self.obstacle3 = np.array([0.75, 0.0, 1.64])

## 3. RL Algorithms

### DDPG and TD3

- **Train**：999 episodes  
![DDPG AND TD3 TRAIN Result](trainresults_1.png)

- **evaluation**：

![DDPG Evaluation Result](ddpg_1.png)

