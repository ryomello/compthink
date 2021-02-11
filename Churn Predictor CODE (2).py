#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries

import numpy as np
import pandas as pd
import random
from pprint import pprint


# In[2]:


# move target variable - churn - to the last column

def order_columns(columns, first_cols=[], last_cols=[], drop_cols=[]):
    columns = list(set(columns) - set(first_cols))
    columns = list(set(columns) - set(drop_cols))
    columns = list(set(columns) - set(last_cols))
    new_order = first_cols + columns + last_cols
    return new_order


# In[3]:


# split the data into train and test

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[4]:


# check partition purity

def check_purity(data):
    
    churn_column = data[:, -1]
    unique_classes = np.unique(churn_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


# In[5]:


# classify if churned or not

def classify_data(data):
    
    churn_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(churn_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# In[6]:


# identify potential splits

def get_potential_splits(data):
    
    global loopCounter
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):          # excluding the last column which is churn
        loopCounter += 1
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


# In[8]:


# split the data

def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    
    # generalisation for continuous variables
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # generalisation for categorical variable  
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above


# In[9]:


#

def calculate_entropy(data):
    
    churn_column = data[:, -1]
    _, counts = np.unique(churn_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


# In[10]:


#

def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy


# In[11]:


#

def determine_best_split(data, potential_splits):
    
    global loopCounter
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            
            loopCounter += 1
  
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# In[12]:


#

def determine_type_of_feature(df):
    
    global loopCounter
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        
        loopCounter += 1
    
        if feature != "Churn":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


# In[13]:


sub_tree = {"question": ["yes_answer", 
                         "no_answer"]}


# In[14]:


# build the tree

def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    
    global loopCounter
    loopCounter += 1
   
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # generalisation for categorical variables
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)    
        
        return sub_tree


# In[15]:


#

def classify_customer(customer, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":  # feature is continuous
        if customer[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(customer[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_customer(customer, residual_tree)


# In[16]:


#

def decision_tree_predictions(df, tree):
    prediction = df.apply(classify_customer, axis=1, args=(tree,))
    
    return prediction


# In[17]:


#

def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_customer, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["Churn"]
    
    accuracy = df["classification_correct"].mean()
    
    return accuracy


# # CHURN PREDICTOR

# In[18]:


# load dataset

df = pd.read_csv (r'C:\Users\sonom\Desktop\UNI\Year 2\Computational Thinking\Final Assignment\TELCO\TELCODATASET.csv')

# reorder columns so that Churn is last

my_list = df.columns.tolist()
reordered_cols = order_columns(my_list, first_cols=[], last_cols=['Churn'], drop_cols=[])
df = df[reordered_cols]

# split data into train and test

train_df, test_df = train_test_split(df, test_size=0.25)

#

data = train_df.values
data = data.astype(np.object)
data[:5]

# build tree on training data

tree = decision_tree_algorithm(train_df, min_samples=2, max_depth=5)

#

customer = test_df.iloc[0]

#

accuracy = calculate_accuracy(test_df, tree)
accuracy


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15, 10]


# In[20]:


# counted loops bigO plot

from timeit import repeat      
from functools import partial  

nvals = [n for n in range(100,3001,300)]         

#times = []
loops = []
for n in nvals:
    ndf = df[:n]              
#    f = partial(decision_tree_algorithm, ndata, counter = 0) 
#    time = min(repeat(f,number=5))/5
#    times.append(time)

    loopCounter = 0
    decision_tree_algorithm(ndf, counter = 0)
    loops.append(loopCounter)

plt.xlabel('length of list (n)')
#plt.ylabel('run time (secs)')
plt.ylabel('loops')
#plt.title('Run Time Big(O) for decision_tree_algorithm')
plt.title('Counted Loops Big Big (O) for decision_tree_algorithm')
#plt.plot(nvals, times, 'o-');
plt.plot(nvals, loops, 'o-');


# In[21]:


# runtime bigO plot

from timeit import repeat      
from functools import partial  

nvals = [n for n in range(100,3001,300)]         

times = []
#loops = []
for n in nvals:
    ndf = df[:n]              
    f = partial(decision_tree_algorithm, ndf, counter = 0) 
    time = min(repeat(f,number=5))/5
    times.append(time)

#    loopCounter = 0
#    decision_tree_algorithm(ndf, counter = 0)
#    loops.append(loopCounter)

#plt.xlabel('length of list (n)')
plt.ylabel('run time (secs)')
#plt.ylabel('loops')
plt.title('Run Time Big(O) for decision_tree_algorithm')
#plt.title('Counted Loops Big Big (O) for decision_tree_algorithm')
plt.plot(nvals, times, 'o-');
#plt.plot(nvals, loops, 'o-');


# In[ ]:


# counted loops bigO plot

from timeit import repeat      
from functools import partial  

nvals = [n for n in range(100,1001,100)]         


loops = []
for n in nvals:
    ndf = df[:n]              

    loopCounter = 0
    decision_tree_algorithm(ndf, counter = 0)
    loops.append(loopCounter)

plt.xlabel('length of list (n)')

plt.ylabel('loops')

plt.title('Counted Loops Big Big (O) for decision_tree_algorithm')

plt.plot(nvals, loops, 'o-');

