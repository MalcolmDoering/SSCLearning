'''
Created on 2023/06/28

@author: malcolm

'''


import tensorflow as tf
import numpy as np



print("tensorflow version", tf.__version__, flush=True)



class SimpleFeedforwardNetwork(object):
    
    def __init__(self, 
                 inputDim, 
                 inputSeqLen, 
                 numOutputClasses,
                 batchSize, 
                 embeddingSize,
                 seed,
                 outputClassWeights):
        
        self.inputDim = inputDim
        self.inputSeqLen = inputSeqLen
        self.numOutputClasses = numOutputClasses
        self.batchSize = batchSize
        self.embeddingSize = embeddingSize
        self.seed = seed
        self.outputClassWeights = outputClassWeights
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        #
        # build the model
        #
        
        # inputs
        self.input_sequences = tf.placeholder(tf.float32, [self.batchSize, self.inputSeqLen, self.inputDim], name="input_sequences")
        
        # targets
        self.shopkeeper_action_id_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="action_id_targets")
        self.shopkeeper_action_id_targets_onehot = tf.one_hot(self.shopkeeper_action_id_targets, self.numOutputClasses)
        
        # class weights
        self.outputClassWeightsTensor = tf.placeholder(tf.float32, [self.numOutputClasses], name="action_weights")

        # mask for not training on bad speech clusters
        self.output_mask = tf.placeholder(tf.int32, [self.batchSize, ], name="output_mask")
        
        
        # input encoding
        with tf.variable_scope("input_encoder"):
            
            # first condense the input vector from each turn (2 layers)
            inputs_reshaped = tf.reshape(self.input_sequences, [self.batchSize*self.inputSeqLen, self.inputDim])
            
            inputs_reshaped_condensed = tf.layers.dense(inputs_reshaped,
                                                        self.embeddingSize,
                                                        activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            inputs_reshaped_condensed = tf.layers.dense(inputs_reshaped_condensed,
                                                        self.embeddingSize,
                                                        activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
                                                        
            inputs_condensed = tf.reshape(inputs_reshaped_condensed, [self.batchSize, self.inputSeqLen, self.embeddingSize])
            
            
            # then feed the sequence of condensed inputs into an two layer feed forward network
            inputs_condensed_reshaped = tf.reshape(self.input_sequences, [self.batchSize, self.inputSeqLen*self.inputDim])
            
            inputs_condensed_reshaped_condensed = tf.layers.dense(inputs_condensed_reshaped,
                                                                  self.embeddingSize,
                                                                  activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            inputs_condensed_reshaped_condensed = tf.layers.dense(inputs_condensed_reshaped,
                                                                  self.embeddingSize,
                                                                  activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            

        # output decoding
        with tf.variable_scope("speech_and_spatial_output/shopkeeper_action_decoder"):
            self.output_action_decoder = tf.layers.dense(inputs_condensed_reshaped_condensed, 
                                                             self.numOutputClasses, 
                                                             activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            
            self.output_action_softmax = tf.nn.softmax(self.output_action_decoder)
            self.output_action_argmax = tf.argmax(self.output_action_softmax, axis=1)
        
        
        with tf.variable_scope("loss"):
            actionLossWeights = tf.squeeze(tf.gather_nd(tf.expand_dims(self.outputClassWeightsTensor, 1), tf.expand_dims(self.shopkeeper_action_id_targets, 1)))
            actionLossWeights = tf.dtypes.cast(self.output_mask, tf.float32) * actionLossWeights
            
            self.output_action_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_action_id_targets_onehot, 
                                                                          self.output_action_decoder,
                                                                          weights=actionLossWeights,
                                                                          reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.loss = self.output_action_loss
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=3e-4)
        #opt = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.09, use_nesterov=True)
        
        self.reset_optimizer_op = tf.variables_initializer(opt.variables())
        
        
        #
        # for training the entire network
        #
        gradients = opt.compute_gradients(self.loss)
        #tf.check_numerics(gradients, "gradients")
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        
        self.train_op_1 = opt.apply_gradients(capped_gradients, name="train_op")
        
        self.train_op = self.train_op_1
        
        
        #
        # setup the prediction function
        #
        self.init_op = tf.initialize_all_variables()
        
        self.initialize()
        
        self.saver = tf.train.Saver()
    
    
    
    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)
        self.sess.run(self.init_op)
    
    
    def get_batches(self, numSamples):
        # note - this will chop off data doesn't factor into a batch of batchSize
        
        batchStartEndIndices = []
        
        for endIndex in range(self.batchSize, numSamples, self.batchSize):
            batchStartEndIndices.append((endIndex-self.batchSize, endIndex))
        
        return batchStartEndIndices
    
    
    def train(self, 
              inputSequenceVectors,
              outputActionIds,
              outputMasks):
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_loss = [] 
        all_shopkeeper_action_loss = [] 
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]], 
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.outputClassWeightsTensor: self.outputClassWeights}
            
            loss, shopkeeper_action_loss,  _ = self.sess.run([self.loss, self.output_action_loss, self.train_op], feed_dict=feedDict)

            ###
            
            all_loss.append(loss) 
            all_shopkeeper_action_loss.append(shopkeeper_action_loss) 
            
            
        return all_loss, all_shopkeeper_action_loss
    
    
    def get_loss(self, 
              inputSequenceVectors,
              outputActionIds,
              outputMasks):
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_loss = [] 
        all_shopkeeper_action_loss = [] 
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]], 
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.outputClassWeightsTensor: self.outputClassWeights}
            
            loss, shopkeeper_action_loss = self.sess.run([self.loss, self.output_action_loss], feed_dict=feedDict)
            
            ###
            
            all_loss.append(loss) 
            all_shopkeeper_action_loss.append(shopkeeper_action_loss) 
            
        return all_loss, all_shopkeeper_action_loss
    
    
    def predict(self, 
              inputSequenceVectors,
              outputActionIds,
              outputMasks):
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_predShkpActionID = [] 
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]], 
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.outputClassWeightsTensor: self.outputClassWeights}
            
            predShkpActionID = self.sess.run(self.output_action_argmax, feedDict)
            
            ###
            
            all_predShkpActionID.append(predShkpActionID)
        
        all_predShkpActionID = np.concatenate(all_predShkpActionID)
        
        
        return all_predShkpActionID
    
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    
    def reset_optimizer(self):    
        self.sess.run(self.reset_optimizer_op)
