from mastermind import *
from DQN import *

mastermind = Mastermind()
stateDim = len(mastermind.color_code)*len(mastermind.colors) + 3*len(mastermind.color_code)
actionDim = len(mastermind.colorPairs)
dqn = DQN(stateDim=stateDim,actionDim=actionDim)

print("Color options: ",mastermind.colors)

print("The correct code:",mastermind.color_code)
guessedCode = random.sample(mastermind.colors,4) # Inital guess is randomized

feedback = mastermind.guess(guessedCode)
print("AI guess:",guessedCode,"Feedback:",feedback)

curState = mastermind.code2state(guessedCode, feedback)

maxAttempt = 20
while (not mastermind.win) and (mastermind.attempts <= maxAttempt):
    # make a guess
    action = dqn.chooseAction(curState, True)
    guessedCode = mastermind.colorPairs[action]
    # Get feedback based on the rules
    feedback = mastermind.guess(guessedCode)
    print("AI guess:", guessedCode, "Feedback:", feedback)
    
    reward = np.sum(feedback)
    # Keep the current state of the game for the network
    nextState = mastermind.code2state(guessedCode, feedback)
    
    curState = nextState
print("Game over,AI attempted %d times"%mastermind.attempts)