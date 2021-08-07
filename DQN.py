import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class FcBlock(nn.Module):
    def __init__(self, inNum, outNum):
        super(FcBlock, self).__init__()
        self.fcConv = nn.Sequential(
            nn.Linear(inNum,outNum),
            nn.BatchNorm1d(num_features=outNum), 
            nn.PReLU(),
        )

    def forward(self, x):
        return self.fcConv(x)

class QvalueNet(nn.Module):
    def __init__(self,stateDim,actionDim):
        super(QvalueNet, self).__init__()
        self.fc1 = FcBlock(stateDim,300)
        self.fc2 = FcBlock(300,300)
        self.fc3 = FcBlock(300, 300)
        self.fc4 = FcBlock(300,actionDim)

    def forward(self, state):
        action = self.fc1(state)
        action = self.fc2(action)
        action = self.fc3(action)
        action = self.fc4(action)
        return action

class DQN():
    def __init__(self,stateDim,actionDim,batchSize=50,memorySize=200,greedRate=0.5,targetRate=0.1,modelPath="model.pkl",lr=1e-4):
        self.stateDim = stateDim 
        self.actionDim = actionDim 
        self.modelPath = modelPath  
        self.batchSize = batchSize 
        self.memorySize = memorySize
        self.greedRate = greedRate
        self.targetRate = targetRate
        self.evalNet = QvalueNet(stateDim,actionDim) 
        self.targetNet = QvalueNet(stateDim, actionDim) 
        self.loadModel()
        self.evalOpt = optim.Adam(self.evalNet.parameters(), lr=lr)
        self.evalCriterion = nn.MSELoss(reduction='sum')

        self.memory = {"curState":[],"action":[],"reward":[],"nextState":[]}

    def saveMemory(self,curState,action,reward,nextState):
        self.memory["curState"].append(curState)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)
        self.memory["nextState"].append(nextState)

        self.memory["curState"] = self.memory["curState"][-1*self.memorySize:]
        self.memory["action"] = self.memory["action"][-1 * self.memorySize:]
        self.memory["reward"] = self.memory["reward"][-1 * self.memorySize:]
        self.memory["nextState"] = self.memory["nextState"][-1 * self.memorySize:]

    def saveModel(self):
        torch.save(self.evalNet.state_dict(),self.modelPath)

    def loadModel(self):
        if os.path.exists(self.modelPath):
            self.evalNet.load_state_dict(torch.load(self.modelPath))
            self.targetNet.load_state_dict(torch.load(self.modelPath))
            print("Model Successfully Loaded")
        else:
            print("Model  does not exist, retraining")

    def updateTargetNet(self):
        self.targetNet.load_state_dict(self.evalNet.state_dict())

    def chooseAction(self, state,greed=False):
        state = np.array(state).reshape(1,-1)
        state = torch.FloatTensor(state)
        self.evalNet.eval()
        qEval = self.evalNet(state)
        if np.random.uniform(0, 1) <= self.greedRate or greed:
            action = np.argmax(qEval.detach().numpy())
            #print('Greed action %s'%action)
        else:
            action = np.random.randint(0, self.actionDim)
            #print('Random action %s' % action)
        return action

    def getBatchMemory(self):
        batchSize = min(self.batchSize, len(self.memory["reward"]))
        indexs = np.random.choice(range(len(self.memory["reward"])),batchSize,replace=False)

        curState = [self.memory["curState"][i] for i in indexs]
        action = [self.memory["action"][i] for i in indexs]
        reward = [self.memory["reward"][i] for i in indexs]
        nextState = [self.memory["nextState"][i] for i in indexs]

        curState = torch.FloatTensor(curState)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        nextState = torch.FloatTensor(nextState)
        return curState,action,reward,nextState

    def learn(self):
        self.evalNet.train()
        curStates, actions, rewards, nextStates = self.getBatchMemory()
        qEval = self.evalNet(curStates)
        qNextTarget = self.targetNet(nextStates)
        qNextTargetAct = torch.argmax(qNextTarget, axis=1)
        qTarget = torch.FloatTensor(qEval.detach().numpy().copy())
        qTarget[np.arange(qTarget.shape[0]),actions] = rewards + self.targetRate*qNextTarget[np.arange(qTarget.shape[0]),qNextTargetAct]
        # Training Network
        loss = self.evalCriterion(qEval, qTarget)
        loss.backward()
        self.evalOpt.step()
        self.evalOpt.zero_grad()
        loss = loss.detach().numpy()
        self.updateTargetNet()
        return loss


if __name__ == "__main__":
    dqn = DQN(100,10)
    state = torch.zeros(1,100)
    action = dqn.chooseAction(state)