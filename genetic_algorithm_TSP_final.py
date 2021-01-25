#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The problem:
# In this assignment, we’ll be using a Genetic Algorithm to find a solution to the traveling salesman problem (TSP). The TSP is described as follows:
# “Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city and returns to the origin city?”
# ![image.png](attachment:image.png)
# Illustration of a potential solution to the Traveling Salesman Problem.

# Given this, there are two important rules to keep in mind:
# - Each city needs to be visited exactly one time
# - We must return to the starting city, so our total distance needs to be calculated accordingly  
# ## Approach:  
# Let’s start with a few definitions, rephrased in the context of the TSP:
# - Gene: a city (represented as (x, y) coordinates)
# - Individual (aka “chromosome”): a single route satisfying the conditions above
# - Population: a collection of possible routes (i.e., collection of individuals)
# - Parents: two routes that are combined to create a new route
# - Mating pool: a collection of parents that are used to create our next population (thus creating the next generation of routes)
# - Fitness: a function that tells us how good each route is (in our case, how short the distance is)
# - Mutation: a way to introduce variation in our population by randomly swapping two cities in a route
# - Elitism: a way to carry the best individuals into the next generation    
#   
# Our GA will proceed in the following steps:
# - 1. Create the population
# - 2. Determine fitness
# - 3. Select the mating pool
# - 4. Breed
# - 5. Mutate
# - 6. Repeat  
# Now, let’s see this in action.

# ## Create necessary classes and functions

# Create two classes: City and Fitness  
# We first create a City class that will allow us to create and handle our cities. These are simply our (x, y) coordinates. Within the City class, we add a distance calculation (making use of the Pythagorean theorem) in line 6 and a cleaner way to output the cities as coordinates with __repr __.

# Create class to handle "cities"

# In[1]:


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


# In[2]:


class City:
    def __init__(self, y, x):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# We’ll also create a Fitness class. In our case, we’ll treat the fitness as the inverse of the route distance. We want to minimize route distance, so a larger fitness score is better. Based on Rule #2, we need to start and end at the same place, so this extra calculation is accounted for in line 13  
# (if i + 1 < len(self.route))  
# of the distance calculation.

# In[3]:


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


# ## Create our initial population

# Route generator.  
# We now can make our initial population (aka first generation). To do so, we need a way to create a function that produces routes that satisfy our conditions (Note: we’ll create our list of cities when we actually run the GA at the end of the tutorial). To create an individual, we randomly select the order in which we visit each city:

# In[4]:


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# Create first "population" (list of routes)  
# This produces one individual, but we want a full population, so let’s do that in our next function. This is as simple as looping through the createRoute function until we have as many routes as we want for our population.

# In[5]:


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


# Note: we only have to use these functions to create the initial population. Subsequent generations will be produced through breeding and mutation.

# ## Create the genetic algorithm - Determine Fitness

# Rank individuals  
# Next, the evolutionary fun begins. To simulate our “survival of the fittest”, we can make use of Fitness to rank each individual in the population. Our output will be an ordered list with the route IDs and each associated fitness score.

# In[6]:


def rankRoutes(population):
    """
    This function sorts the given population in decreasing order of the fitness score.
    """
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


# ## Select the mating pool
# There are a few options for how to select the parents that will be used to create the next generation. The most common approaches are either fitness proportionate selection (aka “roulette wheel selection”) or tournament selection:
# - Fitness proportionate selection (the version implemented below): The fitness of each individual relative to the population is used to assign a probability of selection. Think of this as the fitness-weighted probability of being selected.
# - Tournament selection: A set number of individuals are randomly selected from the population and the one with the highest fitness in the group is chosen as the first parent. This is repeated to chose the second parent.  
# 
# Another design feature to consider is the use of elitism. With elitism, the best performing individuals from the population will automatically carry over to the next generation, ensuring that the most successful individuals persist.  
# For the purpose of clarity, we’ll create the mating pool in two steps. First, we’ll use the output from rankRoutes to determine which routes to select in our selection function. In lines 3–5, we set up the roulette wheel by calculating a relative fitness weight for each individual. In line 9, we compare a randomly drawn number to these weights to select our mating pool. We’ll also want to hold on to our best routes, so we introduce elitism in line 7. Ultimately, the selection function returns a list of route IDs, which we can use to create the mating pool in the matingPool function.

# In[7]:


def selection(popRanked, eliteSize):
    """
    This function takes in a population sorted in decreasing order of fitness score, and chooses a mating pool from it.
    It returns a list of indices of the chosen mating pool in the given population.
    """
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# Now that we have the IDs of the routes that will make up our mating pool from the selection function, we can create the mating pool. We’re simply extracting the selected individuals from our population.

# In[8]:


def matingPool(population, selectionResults):
    """
    This function takes in a population and returns the chosen mating pool which is a subset of the population.
    """
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# ## Breed
# With our mating pool created, we can create the next generation in a process called crossover (aka “breeding”). If our individuals were strings of 0s and 1s and our two rules didn’t apply (e.g., imagine we were deciding whether or not to include a stock in a portfolio), we could simply pick a crossover point and splice the two strings together to produce an offspring.  
# However, the TSP is unique in that we need to include all locations exactly one time. To abide by this rule, we can use a special breeding function called ordered crossover. In ordered crossover, we randomly select a subset of the first parent string (see line 12 in breed function below) and then fill the remainder of the route with the genes from the second parent in the order in which they appear, without duplicating any genes in the selected subset from the first parent (see line 15 in breed function below).
# ![image.png](attachment:image.png)

# In[9]:


def breed(parent1, parent2):
    """
    This function should breed both parents (routes) and return a child route according to the ordered crossover algorithm  
    mentioned above. Please fill in the code to do so.
    """
    # Declaring child 
    child = [] 
    
    # Storing parent info
    parentChild1 = [] 
    parentChild2 = []
    
    # gene 1 & 2
    gene1 = int(random.random() * len(parent1))
    gene2 = int(random.random() * len(parent2))
    
    # Getting the max and min
    startingGene = min(gene1, gene2)
    endingGene = max(gene1, gene2)
    
    # Gets random values
    for i in range(startingGene, endingGene):
        parentChild1.append(parent1[i])
        
    # initilize parentChild2 with values it did not get from parent1
    parentChild2 = [item for item in parent2 if item not in parentChild1]
    
    # initilizing child
    child = parentChild1 + parentChild2
    
    # returns child
    return child


# Next, we’ll generalize this to create our offspring population. In line 5, we use elitism to retain the best routes from the current population. Then, in line 8, we use the breed function to fill out the rest of the next generation.

# In[10]:


def breedPopulation(matingpool, eliteSize):
    """
    This function should return the offspring population from the current population using the breed function. It should 
    retain the eliteSize best routes from the current population. Then it should use the breed function to mate
    members of the population, to fill out the rest of the next generation. You may decide how to choose mates for individuals.
    """
    
    children = []
    size = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    # mate members of population
    for i in range(0, size):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    
    # Returns children population derived from current population using breed function 
    return children


# ## Mutate
# Mutation serves an important function in GA, as it helps to avoid local convergence by introducing novel routes that will allow us to explore other parts of the solution space. Similar to crossover, the TSP has a special consideration when it comes to mutation. Again, if we had a chromosome of 0s and 1s, mutation would simply mean assigning a low probability of a gene changing from 0 to 1, or vice versa (to continue the example from before, a stock that was included in the offspring portfolio is now excluded).  
# However, since we need to abide by our rules, we can’t drop cities. Instead, we’ll use swap mutation. This means that, with a specified low probability, two cities will swap places in our route. We’ll do this for one individual in our mutate function:

# In[11]:


def mutate(individual, mutationRate):
    """
    This function should take in an individual (route) and return a mutated individual. Assume mutationRate is a probability
    between 0 and 1. Use the swap mutation described above to mutate the individual according to the mutationRate. Iterate 
    through each of the cities and swap it with another city according to the given probability.
    """
    #takes in a route and returns mutated version 
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapPlace = int(random.random() * len(individual))
            
            location1 = individual[swapped]
            location2 = individual[swapPlace]
            
            individual[swapped] = location2
            individual[swapPlace] = location1   
    
    # returns mutated individual
    return individual


# Next, we can extend the mutate function to run through the new population.  
# Create function to run mutation over entire population

# In[12]:


def mutatePopulation(population, mutationRate):
    """
    This function should use the above mutate function to mutate each member of the population. Simply iterate over the 
    population and mutate each individual using the mutationRate.
    """
    mutatedPop = []
    # for loop for to itterate through each individual, mutationg using the mutationRate 
    for i in range(0, len(population)):
        mutatedi = mutate(population[i], mutationRate)
        mutatedPop.append(mutatedi)  
   # returns mutatedPop     
    return mutatedPop


# ## Repeat
# We’re almost there. Let’s pull these pieces together to create a function that produces a new generation. First, we rank the routes in the current generation using rankRoutes. We then determine our potential parents by running the selection function, which allows us to create the mating pool using the matingPool function. Finally, we then create our new generation using the breedPopulation function and then applying mutation using the mutatePopulation function.

# In[13]:


def nextGeneration(currentGen, eliteSize, mutationRate):
    """
    This function takes in the current generation, eliteSize and mutationRate and should return the next generation.
    Please use all the above defined functions to do so, some hints are in the above paragraph.
    """ 
    # rank routs
    popRanked = rankRoutes(currentGen)
    # select potential parents 
    selectionResults = selection(popRanked, eliteSize)
    # creating matingPool 
    matingpool = matingPool(currentGen, selectionResults)
    # creating children
    children = breedPopulation(matingpool, eliteSize)
    #applying the mutation 
    nextGeneration = mutatePopulation(children, mutationRate) 
    #returns nextGeneration
    return nextGeneration


# ## Final step: Evolution in motion
# We finally have all the pieces in place to create our GA! All we need to do is create the initial population, and then we can loop through as many generations as we desire. Of course we also want to see the best route and how much we’ve improved, so we capture the initial distance in line 3 (remember, distance is the inverse of the fitness), the final distance in line 8, and the best route in line 9.  

# In[14]:


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    """
    This function creates an initial population, then runs the genetic algorithm according to the given parameters. 
    """
    # declare/initilize variables
    itlPop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(itlPop)[0][1]))
    # loop through diff generation 
    for i in range(0, generations):
        itlPop = nextGeneration(itlPop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(itlPop)[0][1]))
    bestRteIdx = rankRoutes(itlPop)[0][0]
    # initilize return var
    bestRoute = itlPop[bestRteIdx]
    
    # returns bestRoute
    return bestRoute


# ## Running the genetic algorithm

# With everything in place, solving the TSP is as easy as two steps:  
# First, we need a list of cities to travel between. For this demonstration, we’ll use the list of 20 biggest cities of United States (a seemingly small number of cities, but brute force would have to test over 300 sextillion(10^21) routes!):

# ![image.png](attachment:image.png)

# ## Create the City list
# You need to put all the 20 cities as city objects in the cityList array. We will use the last column Coordinates to calculate the distance between all the cities. And for convenience we will multiply all the coordinates by 100. For example, for New York, it would be  
# NewYork= City( int(40.71 * 100), int( -74.00 * 100))  
# And then you can put cityList.append(NewYork)

# Here is the sample code of calculating the path to travel top 5 cities of U.S.

# In[15]:


cityList = []
NewYork= City( int(40.71 * 100), int( -74.00 * 100)) 
LA= City(3405,-11824) 
Chicago=City(4187,-8762)
Houston=City(2976,-9536)
Philly=City(3995,-7516)
cityList.append(NewYork)
cityList.append(LA)
cityList.append(Chicago)
cityList.append(Houston)
cityList.append(Philly)


# Then, running the genetic algorithm is one simple line of code. This is where art meets science; you should see which assumptions work best for you. In this example, we have 100 individuals in each generation, keep 5 elite individuals, use a 1% mutation rate for a given gene, and run through 500 generations:

# In[16]:


geneticAlgorithm(population=cityList, popSize=100, eliteSize=5, mutationRate=0.01, generations=500)


# Therefore, the ideal path to travel is New York - Chicago - LA - Houston - Philly -New York
# Or Chicago - LA - Houston - Philly - New York - Chicago etc.   
# (They are the same, just the starting point is random and could be different)

# ## Visualize the result path

# In[17]:


# Plotting
fig, ax1 = plt.subplots(ncols=1)
y=[4071,4187,3405,2976,3995]
x=[-7400,-8762,-11824,-9536,-7516]
n=['New York','Chicago',"LA","Houston","Philly"]
ax1.plot(x, y, marker="o", markerfacecolor="r")
for i, txt in enumerate(n):
    ax1.annotate(txt, (x[i], y[i]))


# # Your Turn
# Now write the code to calculate the ideal path for traveling the top 20 cities of U.S.

# In[18]:


# Code to create cityList
# Cities and coordinates
cityList = []
NewYork= City( int(40.71 * 100), int( -74.00 * 100)) 
LA= City(3405,-11824) 
Chicago=City(4187,-8762)
Houston=City(2976,-9536)
Philly=City(3995,-7516)
Phoenix=City(3344,-11207)
SanAntonio=City(2942,-9849)
SanDiego=City(3271,-11716)
Dallas=City(3277,-9679)
SanJose=City(3733,-12188)
Austin=City(3026,-9774)
Indianap=City(3976,-8615)
Jacksonville=City(3033,-8165)
SanFrancisco=City(3777,-12241)
Columbus=City(3996,-8299)
Charlotte=City(3522,-8084)
FortWorth=City(3275,-9733)
Detroit=City(4233,-8304)
ElPaso=City(3177,-10644)
Memphis=City(3514,-9004)

cityList.append(NewYork)
cityList.append(LA)
cityList.append(Chicago)
cityList.append(Houston)
cityList.append(Philly)
cityList.append(Phoenix)
cityList.append(SanAntonio)
cityList.append(SanDiego)
cityList.append(Dallas)
cityList.append(SanJose)
cityList.append(Austin)
cityList.append(Indianap)
cityList.append(Jacksonville)
cityList.append(SanFrancisco)
cityList.append(Columbus)
cityList.append(Charlotte)
cityList.append(FortWorth)
cityList.append(Detroit)
cityList.append(ElPaso)
cityList.append(Memphis)


# In[19]:


# Code to run GA
geneticAlgorithm(population=cityList, popSize=100, eliteSize=5, mutationRate=0.01, generations=500) 


# In[25]:


# Code to visulize the result path
# plotting 
fig, ax1 = plt.subplots(ncols=1)
y=[3976,4187,3514,2976,3026,2942,3177,3344,3777,3733,3405,3271,3275,3277,3033,3522,3995,4071,4233,3996]
x=[-8615,-8762,-9004,-9536,-9774,-9849,-10644,-11207,-12241,-12188,-11824,-11716,-9733,-9679,-8165,-8084,-7516,-7400,-8304,-8299]
n=["Indianap","Chicago","Memphis","Houston","Austin","San Antonio","El Paso","Phoenix","San Francisco","San Jose","LA","San Diego","Fort Worth","Dallas","Jacksonville","Charlotte","Philly","New York","Detroit","Columbus"]
ax1.plot(x, y, marker="o", markerfacecolor="r")
for i, txt in enumerate(n):
    ax1.annotate(txt, (x[i], y[i])) 


# # Question: What is the optimal path of cities your GA gave you?

# Answer: The optimal path that was given to me is this: "Indianap","Chicago","Memphis","Houston","Austin","San Antonio","El Paso","Phoenix","San Francisco","San Jose","LA","San Diego","Fort Worth","Dallas","Jacksonville","Charlotte","Philly","New York","Detroit","Columbus"

# ## Plot the progress

# It’s great to know our starting and ending distance and the proposed route, but we would be remiss not to see how our distance improved over time. With a simple tweak to our geneticAlgorithm function, we can store the shortest distance from each generation in a progress list and then plot the results.

# In[21]:


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    """
    This function should be very similar to the geneticAlgorithm function defined above, but it should also create a plot 
    the distance of the best route as a function of how many generations have passed. Please implement it below.
    """
    
    # YOUR CODE HERE
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# Run the GA in the same way as before, but now using the newly created geneticAlgorithmPlot function:

# In[22]:


geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


# ## Conclusion
# I hope this was a fun, hands-on assignment to learn how to build your own GA. Try it for yourself and see how short of a route you can get. Or go further and try to implement a GA on another problem set; see how you would change the breed and mutate functions to handle other types of chromosomes. We’re just scratching the surface here!
