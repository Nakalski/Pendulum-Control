import gym
import random
import numpy as np
import keyboard
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-03
env = gym.make('CartPole-v1')
env.reset()
goal_steps=1000
score_requirement = 120
initial_games = 100

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score=0
        game_memory = []
        previous_observation = []
        for _ in range (goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info  = env.step(action)
            if len(previous_observation) > 0:
                game_memory.append([previous_observation,action])
                
            previous_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    
    print('Average accepted score: ',mean(accepted_scores))
    print('Median accepted score: ',median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network,256, activation='relu', name='PrimeiraCamada')
    network = dropout(network, 0.2)
    network = fully_connected(network,256, activation='relu', name='SegundaCamada')
    network = dropout(network, 0.2)
    network = fully_connected(network,512, activation='relu', name='TerceiraCamada')
    network = dropout(network, 0.2)
    network = fully_connected(network,256, activation='relu', name='QuartaCamada')
    network = dropout(network, 0.2)
    network = fully_connected(network,128, activation='relu', name='QuintaCamada')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax', name='Saida')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network,  tensorboard_verbose=1)
    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=7, snapshot_step=200, show_metric=True, run_id='AndreVinicius')
    return model


with tf.Graph().as_default():
    td=initial_population()
    model = train_model(td)
    model.load('as.model')
    
scores = []
choices = []
env._max_episode_steps = 15000
for each_game in range(1):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    k=0
    for _ in range(env._max_episode_steps):
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        elif (keyboard.is_pressed('a')):
            action = 1 
        elif (keyboard.is_pressed('d')):
            action = 0 
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        choices.append(action)
        
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        #print(k)
        k=k+1
        if done:
            break
        
    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
env.close()
