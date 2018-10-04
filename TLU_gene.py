# This work is partially based on https://gist.github.com/NicolleLouis/d4f88d5bd566298d4279bcb69934f51d

# Only works with python3
import csv
import numpy as np
import random
# import operator
import matplotlib.pyplot as plt


# data formatting
filename = "training-set.csv"
f = open(filename)
reader = csv.reader(f)
data = list(reader)
data = [[float(j) for j in i] for i in data]
samples = np.array(data)[:,0:9]
labels = np.array(data)[:,9]

# parameters
size_population = 1000 # number of weight vectors in GP
dim = 10 # dim of weights
num_copy = int(0.1*size_population) # number of population copied to next gen
num_breeder = int(0.1*size_population)
number_of_child = int((size_population-num_copy)/num_breeder*2)
number_of_generation = 50

#only perceptron() and fitness() use np.array

# the perceptron operation
# weight 1*10ndaray:the weights vector including theta
# sample 1*9ndarray: sample vector
# return threshold result 0/1
def perceptron (weight, sample):
	sample = np.append(sample,[-1]) # append x_10 = -1
	return int(np.sum(sample*weight)>=0)

# test the fitness of the weights vector across all samples
# return percentage of correct outputs
def fitness (weight):
	count = 0
	for row in range(len(samples)):
		sample = samples[row,:]
		if perceptron(np.array(weight), sample)==labels[row]:
			count+=1
			pass
	fitness = count/len(samples)*100
	return fitness

# generate initial weight population, rand [-1,1)
# output: numpy ndarray
def generateFirstPopulation (size_population):
	return (np.random.rand(size_population,dim)*2-1).tolist()

#population: the matrix of weight vector
# return populationSorted
def computePerfPopulation(population):
	populationPerf = []
	for individual in population:
		individual.append(fitness(individual))
		
	population.sort(key=lambda x:x[-1],reverse=True)
	
	for individual in population:
		individual.pop()
	
	return population

# select the individuals copied to the next gen
# and the parents for crossover 	
def selectFromPopulation(populationSorted, num_copy):
	copies = []
	breeders = []
	for i in range(num_copy):
		copies.append(populationSorted[i])
	for i in range(num_copy,num_copy+num_breeder):
		breeders.append(populationSorted[i])
	random.shuffle(breeders)
	
	return [copies,breeders]

# crossover op, randomly choose weight from two weight vectors to form a new vector
def createChild(individual1, individual2):
	child = []
	for i in range(dim):
		if (int(100 * random.random()) < 50):
			child.append(individual1[i])
		else:
			child.append(individual2[i])
	return child

def createChildren(breeders, number_of_child):
	nextPopulation = []
	for i in range(int(len(breeders)/2)):
		for j in range(number_of_child):
			nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
	
	return nextPopulation

# generate the next gen from the last gen
def nextGeneration (firstGeneration, num_copy, number_of_child):
	populationSorted = computePerfPopulation(firstGeneration)
	[copies,nextBreeders] = selectFromPopulation(populationSorted, num_copy)
	nextGeneration = createChildren(nextBreeders, number_of_child)
	nextGeneration=copies+nextGeneration
	return nextGeneration

# gen iteration
def multipleGeneration(number_of_generation, size_population, num_copy, number_of_child):
	historic = []
	historic.append(generateFirstPopulation(size_population))
	for i in range (number_of_generation):
		print("gen "+str(i))
		historic.append(nextGeneration(historic[i], num_copy, number_of_child))
	return historic

#analysis tools
def getBestIndividualFromPopulation (population):
	return computePerfPopulation(population)[0]

def getListBestIndividualFromHistorique (historic):
	bestIndividuals = []
	for population in historic:
		bestIndividuals.append(getBestIndividualFromPopulation(population))
	return bestIndividuals

def printWeightandOutput(individual):
	print("the weight: ")
	print(individual)
	output = []
	for row in range(len(samples)):
		sample = samples[row,:]
		output.append(perceptron(np.array(individual), sample))
	print("output:")
	print(output)

#graph
def evolutionFitness(historic, size_population):
	plt.axis([0,len(historic),0,105])
	plt.title("Evolution Fitness")
	BestFitness=[]
	AvgFitness = []
	for population in historic:
		populationPerf = computePerfPopulation(population)
		BestFitness.append(fitness(populationPerf[0]))
		if population == historic[-1]:
			printWeightandOutput(populationPerf[0])
		averageFitness = 0
		for individual in populationPerf:
			averageFitness += fitness(individual)
		AvgFitness.append(averageFitness/size_population)
	print("Best fitness:")
	print(BestFitness)
	print("Avg fitness:")
	print(AvgFitness)
	plt.plot(BestFitness, label = "Best Fitness")
	plt.plot(AvgFitness, label = "Avg Fitness")
	plt.ylabel('Evolution fitness')
	plt.xlabel('generation')
	plt.legend(loc='lower right')
	plt.savefig("finess.png")

# def evolutionBestFitness(historic):
# 	plt.axis([0,len(historic),0,105])
# 	plt.title("best fitness")
# 	evolutionFitness = []
# 	for population in historic:
# 		evolutionFitness.append(fitness(getBestIndividualFromPopulation(population)))
	
# 	plt.plot(evolutionFitness)
# 	plt.ylabel('fitness best individual')
# 	plt.xlabel('generation')
# 	plt.savefig("bestfiness.png")

# def evolutionAverageFitness(historic, size_population):
# 	plt.axis([0,len(historic),0,105])
# 	plt.title("avgfitness")
# 	evolutionFitness = []
# 	for population in historic:
# 		populationPerf = computePerfPopulation(population)
# 		averageFitness = 0
# 		for individual in populationPerf:
# 			averageFitness += fitness(individual)
# 		evolutionFitness.append(averageFitness/size_population)

# 	plt.plot(evolutionFitness)
# 	plt.ylabel('Average fitness')
# 	plt.xlabel('generation')
# 	plt.savefig("avgfiness2.png")

# main
historic = multipleGeneration(number_of_generation, size_population, num_copy, number_of_child)
evolutionFitness(historic,size_population)
# evolutionAverageFitness(historic, size_population)