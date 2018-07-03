 # coding: utf-8
import numpy as np
import cPickle
import random
import gzip


class Perceptron():
	'''vetor de matrizes de incidências(com pesos nas ligações)
		cada indice do vetor representa uma matriz , onde a k-ésima 
		coluna denota os pesos das ligações entre o k-ésimo neurônio
		com os j-ésimos neuronios(cada um por linha) da camada anterior,
		baseado na notação dos pesos.  
		ex : weights[0][1][2] -> Peso entre a primeira a segunda camada
		,entre o 1 neuronio da primeira camada com o 2 neuronio da segunda camada.
		matriz de bias onde o k-ésimo elemento na l-ésima linha representa o bias
		de um neuronio(neuronio k na linha l)'''
	def __init__(self,layers):
		self.weights = [np.random.randn(j,i) for i,j in zip(layers[:-1],layers[1:])] 
		self.biases = [np.random.randn(i,1) for i in layers[1:]]
		self.num_layers = len(layers)
		self.layers = layers

	'''Função de "achatamento" do produto interno,
	garantindo que a saída final esteja entre 0 e 1'''
	def sigmoid(self,z):
		exp = np.exp(-z)
		return 1.0/(1.0 + exp)

	'''Derivada da função de custo'''	
	def cost_derivative(self,output_activation,target_output):
		return output_activation - target_output

	'''Derivada da função de "achatamento"'''	
	def sigmoid_derivative(self,z):
		return self.sigmoid(z)*(1 - self.sigmoid(z))


	'''Operação básica da rede neural
	computar a saida através de um vetor de entrada
	a = w.a + b --> produto interno + bias'''
	def feedforward(self,input_vector):
		for bias,weight in zip(self.biases,self.weights):
			input_vector = self.sigmoid(np.dot(weight,input_vector) + bias)
		return input_vector

	''' Stochastic Gradient Descent -> Metódo através do qual é possível
	acelear o aprendizado da rede. Separando a entrada de dados em lotes
	(batches), atribuimos um unico custo a cada lote ao invés de um dado
	de entrada individualmente.
	ex:60.000 dados de entrada, lotes de 10,isso acelera o processo em 
	6000 vezes.'''
	def SGD(self,training_data,epoch,mini_batch_size,learning_rate,test_data=None):
		training_data = list(training_data)
		if test_data:
			n_test = len(list(test_data))
		n = len(training_data)
		for i in range(epoch):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update(mini_batch,learning_rate)
			if test_data:
				print("Treino {0}: {1} / {2}".format(i,self.evaluate(test_data),n_test))
			else:
				print("Treino {0} concluído".format(i))

	'''O Método update atualiza todos os peso e biases da rede neural
	usando o retorno da função backprop() que é responsável por calcular
	o gradiente da função de custo
	w" = wk - learning_rate*dC/dw
	b" = bl - learning_rate*dC/db  '''
	def update(self,mini_batch,learning_rate):
		del_b = [np.zeros(b.shape) for b in self.biases]
		del_w = [np.zeros(w.shape) for w in self.weights]
		n = len(mini_batch)
		'''Por estarmos usando o metodo de SGD, criamos um vetor de matrizes
		onde cada matriz tem as respectivas mudanças a serem feitas nos pesos
		e biases'''
		for x,y in mini_batch:
			delta_b, delta_w = self.backprop(x,y)
			del_b = [db + dtb for db,dtb in zip(del_b,delta_b)]
			del_w = [dw + dtw for dw,dtw in zip(del_w,delta_w)]
		''' w" = wk - learning_rate*dC/dw
			b" = bl - learning_rate*dC/db '''
		self.weights = [w - (learning_rate/n)*nw for w,nw in zip(self.weights,del_w)]
		self.biases = [b - (learning_rate/n)*nb for b,nb in zip(self.biases,del_b)]
	
	'''Algoritmo de backpropagation, responsavel por calcular os gradientes
	dC/dw(pesos) e dC/db(bias). A entrada são os dados a serem processados 
	pela rede e as saídas corretas(dizendo quais digitos são por exemplo)
	pois a partir disso é que podemos calcular o erro gerado e assim retornar
	os ajustes a serem feitos nos pesos e bias'''	
	def backprop(self,training_input,target_output):
		del_b = [np.zeros(b.shape) for b in self.biases]
		del_w = [np.zeros(w.shape) for w in self.weights]

		activation = training_input	
		activations = [training_input] # Armazena as ativações de cada camada, pois seram usadas para calcular o gradiente(lista de vetores)
		weighted_sums = [] #lista com as mesmas ativações porém antes de serem "achatadas" pela função sigmoid. ex: w1*a1 + w2*a2...

		for b,w in zip(self.biases,self.weights):
			weighted_sum = np.dot(w,activation) + b
			weighted_sums.append(weighted_sum)
			activation = self.sigmoid(weighted_sum)
			activations.append(activation)
		
		'''Calculando o erro e a respectiva variação na ultima camada, pois a partir desta
		que iremos propagar para o restante da rede'''
		delta = self.cost_derivative(activations[-1],target_output)*self.sigmoid_derivative(weighted_sums[-1])
		del_b[-1] = delta
		del_w[-1] = np.dot(delta,activations[-2].transpose())

		'''loop reverso (back propagation)'''
		for l in range(2,self.num_layers):
			weighted_sum = weighted_sums[-l]
			sig_deriv = self.sigmoid_derivative(weighted_sum)
			delta = np.dot(self.weights[-l+1].transpose(),delta)*sig_deriv
			del_b[-l] = delta
			del_w[-l] = np.dot(delta,activations[-l-1].transpose())
		return (del_b,del_w)
	'''
		Contabiliza a quantidade de saidas corretas '''
	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)



''' Carrega base de dados para a rede neural'''
def load_data():
	with gzip.open('mnist.pkl.gz','rb') as f:
		training_data,validation_data,test_data = cPickle.load(f)
		f.close()
	return (training_data,validation_data,test_data)

def load_data_wrapper():
	tr_d, va_d, te_d = load_data()
	training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
	training_results = [vectorized_result(y) for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, va_d[1])
	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0] ]
	test_data = zip(test_inputs, te_d[1])
	return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


training_data,validation_data,test_data = load_data_wrapper()
net = Perceptron([784,30,10])
net.SGD(training_data,30,10,3.0,test_data=test_data)
