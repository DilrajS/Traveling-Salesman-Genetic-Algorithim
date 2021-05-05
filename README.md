# Traveling-Salesman-Problem (GENETIC ALGORITHIM)

![](images/Screenshot_3.png)

### What is the Traveling Salesman Problem? 

“Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city and returns to the origin city?” 

## What is the Genetic Algorithm? 

A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction to produce offspring of the next generation. 


  **Pros:** 
  * Faster than other algorithms. 
  * Easier. If vector representation of individual is right, we can find out a solution without a deep analysis work.

  **Cons:** 
  * The random heuristics sometimes doesn’t find the optimum.
  * It is not a complete algorithm (not always the algorithm finds a suitable solution). Sometimes it can get stuck with a local maximum problem. However, crossover operation help to mitigate this issue, although this implies more iterations.

### Time Complexity of Genetic Algorithm: 

Genetic Algorithm is an optimization algorithm used to solve non-linear problems, to compute the shortest route in a few seconds (**O(ng)**, where n is population size and g is number of generations). _If this attempted through brute force it would have been impractical with a time complexity of O(n!)._

## Results and Interesting Findings 

**This is the shortest calculated route plotted**

![](images/graph.png) 

**The difference between the initial distance and final distance:**

_Initial distance: 27563.288009700773_

_Final distance: 11593.874222707302_

**Difference**: 15969.41379

### Change in Distance vs Generations

![](images/results.png)

_I did 500 generations but 300 would have been sufficient._

### How to run locally
1.	Go to https://colab.research.google.com (Internet connection and Gmail account required)
2.	Select “GitHub” in the open window and paste in this URL to copy the project 
3.	Confirm you have opened the file ‘Genetic_TSP_notebook_version.ipynb’
4.	Click ‘Runtime’ and select ‘Run all’  

