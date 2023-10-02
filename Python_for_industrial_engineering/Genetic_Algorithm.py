# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:12:02 2021

1. PLEASE DO NOT CHANGE FUNCTION NAMES and PARAMETERS
2. PLEASE ENTER YOUR NAME and STUDENT ID in RESPECTIVE FUNCTIONS
3. PLEASE DO NOT LEAVE ANY CODE OUTSIDE OF THE FUNCTIONS 
4. PLEASE DO NOT TRY READ OR WRITE ANY INPUT/OUTPUT, WORK ONLY WITH THE GIVEN PARAMETERS

@author: ergun
"""
import numpy as np 
import pandas as pd

def getName():
    #TODO: Add your full name instead of Lionel Messi
    return "İbrahim Berk Özkan"

def getStudentID():
    #TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.
    return "070200021"

def generate_initial_solution(df_data, num_elements, max_weight):
    while True:      #if solution is not feasable keep generating
        index=np.random.choice(np.arange(num_elements),int(np.floor(num_elements/3)),replace=False)
        if df_data[index,1].sum()<=max_weight:
            break
    solution=np.zeros(num_elements).astype('int64')
    solution[index]=1 
    return solution

#generates a new solution from existing one by simply toggling (0->1 or 1->0) the value at a random index
def generate_new_sol(df_data,current_sol,num_elements,max_weight):
    # TODO: write your code for generating a new solution
    while True:
        temp_sol=current_sol.copy()
        change_size=np.random.choice([1,2,3],1,p=[0.6,0.3,0.1])
        indexes=np.random.choice(num_elements,change_size,replace=False)
        for index in indexes:
            if temp_sol[index]==1:
                temp_sol[index]=0
            else :
                temp_sol[index]=1
        if calculate_objective(df_data,temp_sol,max_weight)!=0:
            break #temp_sol passed the weight constraint
        
    current_sol=temp_sol
    return current_sol

#calculate the value of the current selection. If the weights exceed the max weigth, this function must return 0 as the value.
def calculate_objective(df_data, solution, max_weight):
    if df_data[solution==1,1].sum()>max_weight:
        return 0
    return df_data[solution==1,0].sum()

def knapsack_simulated_annealing(df_data, max_weight, num_iter = 100, best_obj=False, initial_temp = 10, cooling_rate = 0.99, random_seed = 42):
    np.random.seed(random_seed) # please do not change this line and do not assign a seed again.
    df_data=np.array(df_data) #transform the data to numpy array
    T=initial_temp
    num_elements = df_data.shape[0]
    
    curr_solution=generate_initial_solution(df_data,num_elements,max_weight)
        

    for i in range(num_iter):
        temp_sol=generate_new_sol(df_data,curr_solution,num_elements,max_weight)
        if calculate_objective(df_data,temp_sol,max_weight)>=calculate_objective(df_data,curr_solution,max_weight):
            curr_solution=temp_sol
        else :
            prob=np.exp(-(calculate_objective(df_data,curr_solution,max_weight)-calculate_objective(df_data,temp_sol,max_weight))/T) #by design of my algorithm, curr_sol is always bigger than temp_sol(in this else clause). To make sure prob is between(0,1)
            switch=np.random.choice((1,0),1,p=[prob,1-prob])
#            if i<5 and prob<0.60:
#                initial_temp=initial_temp+10   #if initial temp is not sufficent for given data, bump up litte bit
            if switch==1:
                curr_solution=temp_sol
        T=initial_temp*((cooling_rate)**i) # to make sure prob is always positive  

    if best_obj==True:
        return (curr_solution,calculate_objective(df_data,curr_solution,max_weight)) 
    return curr_solution

# +
dic = {"value": [79,32,47,18,26,85,33,40,45,59],    
           "weights": [85,26,48,21,22,95,43,45,55,52]}

df = pd.DataFrame(dic)
    
max_weight = 300
# -

knapsack_simulated_annealing(df,300,num_iter=300,best_obj=True,initial_temp=10,cooling_rate=0.99,random_seed=42)






