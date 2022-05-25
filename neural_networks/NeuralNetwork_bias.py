import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate, hidden_bias, output_bias):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.hidden_bias = hidden_bias
        self.output_layer_weights = output_layer_weights
        self.output_bias = output_bias

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1/(1 + np.exp(-input))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            # grab the current weights
            cur_weights = [w[i] for w in self.hidden_layer_weights]
            #weighted sum is dot product of current weights and input vector
            weighted_sum = np.dot(inputs,cur_weights)
            #add bias
            weighted_sum += self.hidden_bias[i]
            #now pass through sigmoid activation function
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.
            cur_weights = [w[i] for w in self.output_layer_weights]
            weighted_sum = np.dot(hidden_layer_outputs,cur_weights)
            #add bias
            weighted_sum += self.output_bias[i]
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):
        #print("Doing backprop")
        output_layer_betas = np.zeros(self.num_outputs)
        #print("desired outputs",desired_outputs)
        #print("output layer outputs",output_layer_outputs)
        output_layer_betas = (desired_outputs - output_layer_outputs)
        #print('OL betas: ', output_layer_betas)

        hidden_layer_betas = np.zeros(self.num_hidden)
        for j in range(self.num_hidden):
            beta = 0
            cur_weights = self.output_layer_weights[j]
            for k in range(self.num_outputs):
                o_k = output_layer_outputs[k]
                beta_k = output_layer_betas[k]
                beta += cur_weights[k]*o_k*(1 - o_k)*beta_k
            hidden_layer_betas[j] = beta
        #print('HL betas: ', hidden_layer_betas)


        # Formula for weights:
        # delta w_{i,j} = eta * o_i * o_j * (1 - o_j) * beta_j

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                o_i = hidden_layer_outputs[i]
                o_j = output_layer_outputs[j]
                beta_j = output_layer_betas[j]
                eta = self.learning_rate
                delta_output_layer_weights[i][j] = eta * o_i * o_j * (1 - o_j) * beta_j

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
               o_i = inputs[i]
               o_j = hidden_layer_outputs[j]
               beta_j = hidden_layer_betas[j]
               eta = self.learning_rate
               delta_hidden_layer_weights[i][j] = eta * o_i * o_j * (1 - o_j) * beta_j

        #print("Output layer deltas", delta_output_layer_weights)
        #print("Hidden layer deltas", delta_hidden_layer_weights)

        delta_hidden_bias = []
        #updating hidden bias
        for i in range(self.num_hidden):
            o_i = hidden_layer_outputs[i]
            beta_i = hidden_layer_betas[i]
            eta = self.learning_rate
            delta_b_i = eta * o_i * (1 - o_i) * beta_i
            delta_hidden_bias.append(delta_b_i)

        delta_output_bias=[]
        #updating output bias
        for i in range(self.num_outputs):
            o_i = output_layer_outputs[i]
            beta_i = output_layer_betas[i]
            eta = self.learning_rate
            delta_b_i = eta * o_i * (1 - o_i) * beta_i
            delta_output_bias.append(delta_b_i)

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights, delta_hidden_bias, delta_output_bias

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights, delta_hidden_bias, delta_output_bias):
        # TODO! Update the weights.
        self.output_layer_weights += delta_output_layer_weights
        self.hidden_layer_weights += delta_hidden_layer_weights
        
        self.hidden_bias += delta_hidden_bias
        self.output_bias += delta_output_bias

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, delta_hidden_bias, delta_output_bias = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = np.argmax(output_layer_outputs) 
                predictions.append(predicted_class)
                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights, delta_hidden_bias, delta_output_bias)

            # Print new weights
            #print('Hidden layer weights \n', self.hidden_layer_weights)
            #print('Output layer weights  \n', self.output_layer_weights)

            # TODO: Print accuracy achieved over this epoch
            total_right = 0
            total = len(predictions)
            #print("predictions", predictions)
            #print("desired outputs", desired_outputs)
            for i in range(total):
                pred = predictions[i]
                real = np.argmax(desired_outputs[i])
                #print(f"guess was {pred}, real {real}")
                if pred == real:
                    total_right += 1
            acc = total_right / total
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            print(f"hidden layer output: {hidden_layer_outputs}")
            print(f"output layer outputs: {output_layer_outputs}")
            predicted_class = np.argmax(output_layer_outputs)  # TODO! Should be 0, 1, or 2.
            predictions.append(predicted_class)
        return predictions
