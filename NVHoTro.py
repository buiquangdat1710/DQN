import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam, Adamax
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
from flask import Flask, request
import requests
import gc 
tf.config.list_physical_devices('GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


LOAD_MODEL = "models/Best_Model_in_Epochs95_Reward1755.model"

ACTION_SPACE_SIZE = 4
OBSERVATION_SPACE_VALUES = (135,240, 1)
TIME_SLEEP = 0.05
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # Kích cỡ tối đa của replay_buffer
MIN_REPLAY_MEMORY_SIZE = 10  # Kích cỡ tối thiểu (nếu lớn hơn thì lấy sample ra để training)
MINIBATCH_SIZE = 3  # Kích cỡ batch để training
UPDATE_TARGET_EVERY = 5  # Số bước để update target model
MODEL_NAME = 'Tang1' # tên model để lưu
MIN_REWARD = 1400  # Reward lớn hơn thì lưu model vào

ID_IMAGE = 2
ID_MODEL = 98

EPISODES = 100 # Số lần chơi

# Exploration settings
epsilon = 0.7
EPSILON_DECAY = 0.995 # giảm epsilon sau mỗi bước
MIN_EPSILON = 0.001 # min épilon

#  Stats settings
AGGREGATE_STATS_EVERY = 2  # episodes show
SHOW_PREVIEW = True

class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()




if not os.path.isdir('models'):
    os.makedirs('models')

class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Biến này dùng để update target model
        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")
        else:
            
            model = Sequential()
            model.add(Conv2D(256, (3, 3), input_shape=OBSERVATION_SPACE_VALUES))  
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())  
            model.add(Dense(64))
            model.add(BatchNormalization())
            model.add(Dense(ACTION_SPACE_SIZE, activation='linear')) 
            model.compile(loss="mse", optimizer = Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999,  clipvalue = 1.0), metrics=['mae'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = []
        for state in current_states:
            current_qs_list.append(self.model.predict(state,verbose = 0))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = []
        for state in new_current_states :
            future_qs_list.append(self.model.predict(state,verbose = 0))

        X = [] # data
        y = [] # label

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs = current_qs.flatten()
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        X = np.array(X).reshape(-1, 135, 240, 1)
        y = np.array(y).reshape(-1, ACTION_SPACE_SIZE)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(X/255, y, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Nếu done = True thì target_update_counter tăng lên 1
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state/255,verbose = 0)[0]
    
agent = DQNAgent()

def STATE():
    global ID_IMAGE
    while not os.path.exists("received_image" + str(ID_IMAGE) + ".png"):
        time.sleep(TIME_SLEEP)
        # print("Ko nhận được ảnh")
    while not os.path.exists("received_image" + str(ID_IMAGE) + ".png"):
        time.sleep(TIME_SLEEP)
    current_state = cv2.imread("received_image" + str(ID_IMAGE) + ".png")
    # print("ID ảnh: ",ID_IMAGE)
    current_state = cv2.resize(current_state, (240, 135), interpolation=cv2.INTER_AREA)
    current_state = cv2.cvtColor(current_state, cv2.COLOR_BGR2GRAY)
    current_state = np.array(current_state )
    current_state = np.expand_dims(current_state , axis=-1)
    current_state  = np.expand_dims(current_state , axis= 0 )

    # Đọc giá trị reward, done
    done = 0
    reward = -1
    with open("Reward.txt", "r") as file:
        reward = int(file.read())
    with open("Done.txt", "r") as file:
        done = int(file.read())

    os.remove("received_image" + str(ID_IMAGE) + ".png")
    ID_IMAGE = ID_IMAGE + 1

    gc.collect()
    return current_state, reward, done
def random_action():
    # 40,20,20,20
    rand = random.random()
    if rand < 0.4:
        return 0 # sang trai
    elif rand < 0.6:
        return 1 # nhay len
    elif rand < 0.8:
        return 2 # sang phai
    else:
        return 3 # danh


ep_rewards = []



with tf.device('/GPU:0'):
    for episode in range(1,EPISODES + 1):
        response = requests.post("http://localhost:8000/data", data= "1")
        action_list = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Lúc vào Game (ở dạng array (864, 1536, 3))
        current_state, reward, done = STATE() # 2 
        while not done:
            # print("Done: ",done)
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state)) 
            else:
                # Get random action
                action = random_action()
            
            action_list.append(action)
            data = str(action)
            response = requests.post("http://localhost:8000/data", data=data)
            # SAU KHI CÓ ACTION RỒI THÌ GỬI ACTION QUA BE RỒI BE GỬI NEW_STATE, REWARD, DONE
            # GỬI ACTION QUA BE:
            #.................
            #.................
            # NHẬN NEW_STATE, REWARD, DONE

            new_state, reward, done = STATE() # 3 
            # print("Reward: ",reward)


            # Transform new continous state to new discrete state and count reward
            episode_reward += reward


            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1
        print("Đã train xong Epochs: ",episode)
        print("Tổng Reward sau 1 epochs: ",episode_reward)
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if episode_reward >= MIN_REWARD:
            with open("Actiontang2.txt", "a") as file:
                file.write(' '.join(map(str,action_list)) + '\n')
            agent.model.save(f'models/Best_Model_in_Epochs{episode + ID_MODEL}_Reward{episode_reward}.model')
        else:
            agent.model.save(f'models/{MODEL_NAME}_Epochs{episode + ID_MODEL}_Reward{episode_reward}.model')
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
agent.model.save(f'models/{MODEL_NAME}_final.model')