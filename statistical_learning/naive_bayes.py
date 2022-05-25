import pandas as pd
import numpy as np

def grab_data(filename):
    """grabs data from a csv file
       returns a pandas dataframe
       as well as label data
    """
    df = pd.read_csv(filename)
    #pop index column, pandas already has one
    df = df.iloc[:,1:]
    labels = df.pop('class')
    return [df,labels]

def init_table(df, labels):
    """ build the empty probability table
    table is a nested dictionary 
    that has the following structure for each feature
    table = {
            "feature_1":{
                "value_1":{
                    "no-recurrence-events":1
                    "recurrence-events":1
                },"value_2":{
                    "no-recurrence-events":1
                    "recurrence-events":1
                }
            }
    To see the number of recurrence events for feature_1 with value_1:
    table["feature_1"]["value_1"]["recurrence-events"]
    """
    table = {}
    keys = list(df)
    for key in keys:
        table[key] = {}
        values = list(df[key])
        unique_values = list(np.unique(values))
        for val in unique_values:
            table[key][val] = {"no-recurrence-events":1, "recurrence-events":1}
    return table

def fill_table(df,labels,table):
    #iterate over instance
    #slow with pandas but our dataset is small
    #and code stays readable
    for i in range(df.shape[0]):
        #grab instance
        row=df.iloc[i,:]
        #class label
        label=labels.iloc[i]
        #iterate over features
        for key in row.keys():
            value = row[key]
            #add to table count
            before=table[key][value]
            #print(f"table[{key}][{value}] = {before}")
            #print(f"class is {label}, updating count")
            if label=="recurrence-events":
                table[key][value]["recurrence-events"] += 1
            else:
                table[key][value]["no-recurrence-events"] += 1
            after=table[key][value]
    return table

def train(df,labels,table):
    """ calculate probability table """
    prob=init_table(df,labels)
    #relative probability of each class
    num_recur = labels[labels=='recurrence-events'].shape[0]
    num_no_recur = df.shape[0] - num_recur
    prob['recurrence-events']=num_recur
    prob['no-recurrence-events']=num_no_recur

    #calculate probabilities
    for key in df.keys():
        values = list(df[key])
        unique_values = list(np.unique(values))
        for val in unique_values:
            p_recur = table[key][val]['recurrence-events'] / num_recur
            p_no_recur = table[key][val]['no-recurrence-events'] / num_no_recur
            prob[key][val]['recurrence-events'] = p_recur
            prob[key][val]['no-recurrence-events'] = p_no_recur
    return prob

def class_score(instance, class_label, prob):
    """ calculate a class score given an instance, 
        class label and a probability table. """
    score = prob[class_label]
    for feature in instance.keys():
        value = instance[feature]
        score = score * prob[feature][value][class_label]
    return score

def test(prob,filename):
    #test on our filename given our probability table
    print("testing on file ",filename)
    df,labels = grab_data(filename)
    total_right = 0
    total=df.shape[0]
    preds = []
    for i in range(df.shape[0]):
        row = df.iloc[i,:]
        print("Prediction for the following instance:")
        print(row)
        scores={'recurrence-events':0,'no-recurrence-events':0}
        #calculate scores for a given class label
        for class_label in scores.keys():
            scr = class_score(row,class_label,prob)
            scores[class_label] = scr
        print("Scores: ")
        print(scores)
        if scores['recurrence-events'] > scores['no-recurrence-events']:
            pred = 'recurrence-events'
        else:
            pred = 'no-recurrence-events'
        print(f"I guessed {pred}. Actual class is {labels.iloc[i]}")
        if pred==labels.iloc[i]:
            print("Correctly guessed.")
            total_right += 1
        preds.append(pred)
    acc = total_right / total
    print("Accuracy: ",acc)
    print("Predictions: ",preds)
    print("Actual: ",labels)
   
def print_section(str):
    print()
    print("-"*30)
    print(str)
    print("-"*30)
    print()

if __name__=="__main__":
    print_section("Starting Classifier...")
    print_section("Grabbing data.")
    [df,labels]=grab_data("breast-cancer-training.csv")
    table=init_table(df,labels)
    table=fill_table(df,labels,table)
    print_section("Training.")
    prob=train(df,labels,table)
    print("Probability table:")
    for key in prob.keys():
        print(key)
        try:
            for nested_key in prob[key].keys():
                print(" "*4,nested_key)
                print(" "*8,prob[key][nested_key])
        except:
            print(prob[key])
    print_section("Testing.")
    test(prob,"breast-cancer-test.csv")
