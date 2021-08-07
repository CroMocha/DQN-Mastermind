from mastermind import *
from DQN import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

mastermind = Mastermind()
stateDim = len(mastermind.color_code)*len(mastermind.colors) + 3*len(mastermind.color_code)
actionDim = len(mastermind.colorPairs)
dqn = DQN(stateDim=stateDim,actionDim=actionDim)
playCount = 0
maxAttempt = 20
lossHistory = []
history = {"play count":[],"attempt":[],"loss":[]}

while True:
    mastermind.reset()

    prevGuesses = []

    guessedCode = random.sample(mastermind.colors,4)  # Inital guess is randomized
 
    feedback = mastermind.guess(guessedCode)

    curState = mastermind.code2state(guessedCode,feedback)

    # Training
    while (not mastermind.win) and (mastermind.attempts<=maxAttempt):
        # make a guess
        action = dqn.chooseAction(curState,False)

        guessedCode = mastermind.colorPairs[action]
        # Get feedback based on the rules
        feedback = mastermind.guess(guessedCode)

        reward = np.sum(feedback)

        # penalty for retrying the wrong guess
        if guessedCode in prevGuesses: 
            reward = 0

        prevGuesses.append(guessedCode)
        # Keep the current state of the game for the network
        nextState = mastermind.code2state(guessedCode,feedback)

        dqn.saveMemory(curState,action,reward,nextState)

        curState = nextState
    
    # Learning
    loss = dqn.learn()
    lossHistory.append(loss)
    playCount += 1

    # Testing and Logging
    if playCount%1000 == 0:
        attCounts = []
        testCount = 0
        while testCount<100:
            testCount += 1
            mastermind.reset()
            
            guessedCode = random.sample(mastermind.colors,4)
            
            feedback = mastermind.guess(guessedCode)
            
            curState = mastermind.code2state(guessedCode, feedback)
            
            while (not mastermind.win) and (mastermind.attempts<=maxAttempt):
                # make a guess
                action = dqn.chooseAction(curState,True)
                
                guessedCode = mastermind.colorPairs[action]
                # Get feedback based on the rules
                feedback = mastermind.guess(guessedCode)
                
                reward = np.sum(feedback)
                # Keep the current state of the game for the network
                nextState = mastermind.code2state(guessedCode,feedback)
                
                curState = nextState
            attCounts.append(mastermind.attempts)
        print("Play:%d Attempt:%d Loss:%.3f"%(playCount,np.mean(attCounts),np.mean(lossHistory[-100:])))
        dqn.saveModel()

        # Visualisation
        history["play count"].append(playCount)
        history["attempt"].append(np.mean(attCounts))
        history["loss"].append(np.mean(lossHistory[-100:]))
        historyDF = pd.DataFrame(history)
        historyDF.to_csv("train_process.csv",index=None)

        fig, ax = plt.subplots(1, 1)
        ax.plot(history["play count"],history["attempt"],color="k")
        ax.set_xlabel("play count")
        ax.set_ylabel("attempt")
        fig.savefig("train_attempt.jpg",dpi=500)

        fig, ax = plt.subplots(1, 1)
        ax.plot(history["play count"],history["loss"],color="k")
        ax.set_xlabel("play count")
        ax.set_ylabel("loss")
        fig.savefig("train_loss.jpg",dpi=500)