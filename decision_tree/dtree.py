import pandas as pd
import numpy as np
import argparse 

#Need to change these
#depending on dataset.

#For hepatitis dataset
COND_TRUE="live"
COND_FALSE="die"

#for golf dataset
"""
COND_TRUE="PlayGolf"
COND_FALSE="StayHome"
"""

ATT_TRUE=True
ATT_FALSE=False
MOST_LIKELY_OUTCOME=None

def read_data(filename):
    """ Auxilliary function to read the data in initially """
    #read from csv
    #whitespace delimiter
    df = pd.read_csv(filename, delim_whitespace=True)
    attributes = list(df.columns.values)
    #pop 'CLASS' as this is our
    #results column
    attributes.pop(0)
    #gross global variable access but whatever
    num_true=len(df[(df['Class']==COND_TRUE)])
    num_false=len(df[(df['Class']==COND_FALSE)])
    global MOST_LIKELY_OUTCOME
    if num_true >= num_false:
        MOST_LIKELY_OUTCOME=COND_TRUE
    else:
        MOST_LIKELY_OUTCOME=COND_FALSE
    return [df, attributes]

def cross_validate(train_filenames,test_filenames):
    #cross validate on filenames
    #code to run training and testing
    #what k-fold validation are we doing
    k_num = len(train_filenames)
    #accuracy array
    accs=[]
    for i in range(k_num):
        train_file = train_filenames[i]
        test_file = test_filenames[i]
        [df, attributes] = read_data(train_file)
        dtree=d_tree(df, attributes)
        dtree.report()
        acc = dtree.test_file(test_file)  
        accs.append(acc)
    print("accuracy array: {}".format(accs))
    print("{} fold cross validation accuracy: {}".format(k_num,np.mean(accs)))

""" Given a set of instances (a data frame with training samples)
and a list of attributes of those samples, build a tree.
The tree node has an attribute, a left tree, and a right tree. 
For a leaf node, the tree has no left tree or right tree 
(these stay as type None)"""
class d_tree:

    def __init__(self,df,attributes):
        #is this a leaf node? (I.e do we have an outcome?)
        self.is_leaf = False
        self.left_tree = None
        self.right_tree = None
        self.best_att = None
        #depth for tree traversal
        [self.df, self.attributes] = [df,attributes]
        #build the tree out.
        self.build_tree()

    def test_file(self, filename):
        """Test accuracy on a file"""
        df = pd.read_csv(filename, delim_whitespace=True)
        [rows,cols] = df.shape
        
        total = rows
        total_right = 0
        total_right_baseline = 0
        for i in range(rows):
            test_class = self.test(df.iloc[i])
            if test_class == df['Class'].iloc[i]:
                total_right += 1
            if MOST_LIKELY_OUTCOME == df['Class'].iloc[i]:
                total_right_baseline += 1
        accuracy = (total_right/total) * 100
        baseline_acc = (total_right_baseline / total) * 100
        print("Got {}/{} right.".format(total_right,total))
        print("Accuracy: {} %".format(accuracy))
        print("Baseline accuracy: {} %".format(baseline_acc))
        return accuracy

    def test(self, test_sample):
        """Test the decision tree on a specific instance"""
        #start with root note
        cur_tree = self
        while True:
            print("Visiting {}".format(cur_tree.best_att))
            if cur_tree.is_leaf:
                return cur_tree.best_att
            if test_sample[cur_tree.best_att]:
                #True, go down left branch
                cur_tree=cur_tree.left_tree
            else:
                #False, go down right branch
                cur_tree=cur_tree.right_tree

    def report(self, indent=""):
        """Print the tree out."""
        if self.is_leaf:
            print("{}Class {}".format(indent, self.best_att))
        else:
            print("{}{} = True:".format(indent, self.best_att))
            self.left_tree.report(indent+"    ")
            print("{}{} = False:".format(indent, self.best_att))
            self.right_tree.report(indent+"    ")

    def get_impurity(self,attribute):
        """Calculate the impurity of an atttribute."""
        print("calculating impurity for attribute {}".format(attribute))
        #result when our attribute is True
        has_att = self.df['Class'][(self.df[attribute]==ATT_TRUE)]
        print("has attribute: {}".format(has_att))
        #result when our attribute isn't True
        no_att = self.df['Class'][(self.df[attribute]==ATT_FALSE)]
        print("no attribute: {}".format(no_att))

        #total number of samples
        total_num=self.df.shape[0]
        #calculate weighted impurity score
       
        #weight for True (has attribute)
        w_t = len(has_att)/total_num
        #weight for False (doesn't have attribute)
        w_f = len(no_att)/total_num
        has_att_play = len(has_att[(self.df['Class']==COND_TRUE)])
        #print("has att, plays {}".format(has_att_play))
        has_att_noplay = len(has_att[(self.df['Class']==COND_FALSE)])
        #print("has att, no play {}".format(has_att_noplay))
        no_att_play = len(no_att[(self.df['Class']==COND_TRUE)])
        #print("no att, plays {}".format(no_att_play))
        no_att_noplay = len(no_att[(self.df['Class']==COND_FALSE)])
        #print("no att, no play {}".format(no_att_noplay))

        #default to 0 weight to catch divide by zero error
        if (has_att_play==0) or (has_att_noplay==0):
            impurity_has_att=0
        else:
            impurity_has_att = (has_att_play * has_att_noplay) / (len(has_att)**2)
        if (no_att_play==0) or (no_att_noplay==0):
            impurity_no_att=0
        else:
            impurity_no_att = (no_att_play * no_att_noplay) / (len(no_att)**2)

        print("true calculation:")
        print("{}/{} * {}/{} * {}/{} = {}".format(len(has_att),total_num,has_att_play,len(has_att),has_att_noplay,len(has_att),impurity_has_att*w_t))

        print("false calculation:")
        print("{}/{} * {}/{} * {}/{} = {}".format(len(no_att),total_num,no_att_play,len(no_att),no_att_noplay,len(no_att),impurity_no_att*w_f))
        

        impurity = w_t * impurity_has_att + w_f * impurity_no_att

        #dataframe subsets: attribute true and attribute false
        att_true = self.df[(self.df[attribute]==ATT_TRUE)]
        att_false = self.df[(self.df[attribute]==ATT_FALSE)]
        return [att_true, att_false, impurity]

    def is_pure(self):
        """Determine if an attribute is true"""
        #return true and class if the set of instances is 100% pure.
        #all true?
        df = self.df
        assert df.empty!=True, "Error, df is empty."
        num_true=len(df[(df['Class']==COND_TRUE)])
        num_false=len(df[(df['Class']==COND_FALSE)])
        num=len(df)
        if (num==num_true):
            return [True,COND_TRUE]
        elif (num==num_false):
            return [True,COND_FALSE]
        else:
            return [False,None]

    def build_tree(self):
        """Build the tree using some training data"""
        print("building tree.")
        #print("data frame: {}".format(self.df))
        #print("attributes: {}".format(self.attributes))
        #case 1: instances is empty
        if self.df.empty:
            print("case 1: empty instances")
            self.is_leaf = True
            self.best_att = MOST_LIKELY_OUTCOME
        #case 2: instances are pure
        #is_pure returns [True/False,Class/None]
        elif self.is_pure()[0]:
            print("case 2: pure instances")
            self.is_leaf = True
            self.best_att = self.is_pure()[1]
        #case 3: no attributes
        elif len(self.attributes)==0:
            print("case 3: no attributes")
            #return most likely class
            if (len(self.df['Class']==COND_TRUE) >= 
                len(self.df['Class']==COND_FALSE)):
                self.best_att = COND_TRUE
            else:
                self.best_att = COND_FALSE
            self.is_leaf=True
        #case 4: gotta find best attribute
        else:
            print("case 4: finding best attribute")
            #grab impurities
            #get_impurity returns [att_true, att_false, impurity]
            impurities=[self.get_impurity(att)[2] for att in self.attributes]
            #show impurities
            for (i,att) in enumerate(self.attributes):
                imp = impurities[i]
                print("attribute {} has impurity {}".format(att,imp))
            ##print most pure
            pure_ind = np.argmin(impurities)
            #set best attribute to most pure
            self.best_att = self.attributes[pure_ind]
            #grab att_true and att_false
            [att_true, att_false, imp] = self.get_impurity(self.best_att)
            #pop best attriibute from list of atts
            self.attributes.remove(self.best_att)
            #initialise left and right tree
            #left branch has attribute, right branch doesn't.
            self.left_tree=d_tree(att_true,self.attributes)
            self.right_tree=d_tree(att_false,self.attributes)

if __name__=="__main__":
    """
    Read from command line training file and testing
    file.
    arguments:
    training file (string)
    --train hep_train.txt
    testing file (string)
    --test hep.txt
    
    Example:
    python3 dtree.py --train trainfile.txt --test testfile.txt
    """

    #command line argument parser stuff
    parser = argparse.ArgumentParser()
    parser.add_argument('-cross', '--xvalidate',
                        help='Usage: --xvalidate 1',
                        type=bool
                       )
    parser.add_argument('-train', '--train_file',
                        help='Usage: -train train_file.txt',
                        type=str,
                        )
    parser.add_argument('-test', '--test_file',
                        help='Usage: -test test_file.txt',
                        type=str,
                        )
    #this file expects all the hepatitis files from Datasets
    #grab cmd line arguments from parser
    args = parser.parse_args()
    if args.xvalidate == 1:
        print("running x-validation script.")
        test_filenames=["hepatitis-test-run-{}.txt".format(i) for i in range(10)]
        train_filenames=["hepatitis-training-run-{}.txt".format(i) for i in range(10)]
        cross_validate(test_filenames,train_filenames)
    else:
        assert args.train_file is not None, "Error: specify a training file."
        assert args.test_file is not None, "Error: specify a testing file."

        #code to run training and testing
        [df, attributes] = read_data(args.train_file)
        dtree=d_tree(df, attributes)
        dtree.report()
        dtree.test_file(args.test_file)
