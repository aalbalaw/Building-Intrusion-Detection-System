
# coding: utf-8

# In[1]:

## Preliminaries
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing,cross_validation, feature_extraction
from sklearn import linear_model, svm, metrics, ensemble, tree, ensemble
from sklearn.decomposition import PCA
import pandas as pd
import urllib
import csv

# Helper functions
def folds_to_split(data,targets,train,test):
    data_tr = pd.DataFrame(data).iloc[train]
    data_te = pd.DataFrame(data).iloc[test]
    labels_tr = pd.DataFrame(targets).iloc[train]
    labels_te = pd.DataFrame(targets).iloc[test]
    return [data_tr, data_te, labels_tr, labels_te]


# 
# 

# ## Using SVM To Build The Model 5 class labeling:

# In[22]:

#let's load the data
train_data = urllib.urlopen('/home/aziz/Downloads/kddcup.data_10_percent_corrected')
test_data = urllib.urlopen('/home/aziz/Downloads/corrected')

#Place both dataset into a dataframe
train_multiclass = pd.read_csv(train_data, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])
test_multiclass = pd.read_csv(test_data, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])


# 
# 

# ## 1 Pre-Processing The Datasets:

# ### 1.1 Change Labels to The Right Class: 

# In[23]:

## Replacing all the different attack types(24) to their proper general attack class
train_multiclass.loc[(train_multiclass['Class'] =='smurf.')|(train_multiclass['Class'] =='neptune.') | (train_multiclass['Class'] =='back.') | (train_multiclass['Class'] =='teardrop.') |(train_multiclass['Class'] =='pod.')| (train_multiclass['Class']=='land.'),'Class'] = 'Dos'
train_multiclass.loc[(train_multiclass['Class'] =='satan.')|(train_multiclass['Class'] =='ipsweep.') | (train_multiclass['Class'] =='portsweep.') | (train_multiclass['Class'] =='nmap.'),'Class'] = 'probe'
train_multiclass.loc[(train_multiclass['Class'] =='spy.')|(train_multiclass['Class'] =='phf.')|(train_multiclass['Class'] =='multihop.')|(train_multiclass['Class'] =='ftp_write.') | (train_multiclass['Class'] =='imap.') | (train_multiclass['Class'] =='warezmaster.') |(train_multiclass['Class'] =='guess_passwd.')| (train_multiclass['Class']=='warezclient.'),'Class'] = 'r2l'
train_multiclass.loc[(train_multiclass['Class'] =='buffer_overflow.')|(train_multiclass['Class'] =='rootkit.') | (train_multiclass['Class'] =='loadmodule.') | (train_multiclass['Class'] =='perl.'),'Class']='u2r'
train_multiclass.loc[(train_multiclass['Class'] =='normal.'),'Class'] = 'normal'


# 
# 

# In[24]:

## Replacing all the different attack types(36) to their proper general attack class


test_multiclass.loc[(test_multiclass['Class'] =='smurf.')|(test_multiclass['Class'] =='neptune.') | 
                    (test_multiclass['Class'] =='back.') | (test_multiclass['Class'] =='teardrop.') |
                    (test_multiclass['Class'] =='pod.')| (test_multiclass['Class']=='land.')|
                   (test_multiclass['Class']=='apache2.')|(test_multiclass['Class']=='udpstorm.')|
                   (test_multiclass['Class']=='processtable.')|(test_multiclass['Class']=='mailbomb.'),'Class'] = 'Dos'


test_multiclass.loc[(test_multiclass['Class'] =='guess_passwd.')|(test_multiclass['Class'] =='ftp_write.')|
                    (test_multiclass['Class'] =='imap.')|(test_multiclass['Class'] =='phf.') | 
                    (test_multiclass['Class'] =='multihop.') | 
                    (test_multiclass['Class'] =='warezmaster.') |(test_multiclass['Class'] =='snmpgetattack.')| 
                    (test_multiclass['Class']=='named.')|(test_multiclass['Class'] =='xlock.')|
                    (test_multiclass['Class'] =='xsnoop.')|(test_multiclass['Class'] =='sendmail.')|
                    (test_multiclass['Class'] =='httptunnel.')|(test_multiclass['Class'] =='worm.')|
                    (test_multiclass['Class'] =='snmpguess.'),'Class'] = 'r2l'

test_multiclass.loc[(test_multiclass['Class'] =='satan.')|(test_multiclass['Class'] =='ipsweep.') | (test_multiclass['Class'] =='portsweep.') | (test_multiclass['Class'] =='nmap.')|
                    (test_multiclass['Class'] =='saint.')|(test_multiclass['Class'] =='mscan.'),'Class'] = 'probe'

test_multiclass.loc[(test_multiclass['Class'] =='buffer_overflow.')|(test_multiclass['Class'] =='rootkit.') | 
                    (test_multiclass['Class'] =='loadmodule.') | (test_multiclass['Class'] =='xterm.')|
                    (test_multiclass['Class'] =='sqlattack.')|(test_multiclass['Class'] =='ps.')|
                    (test_multiclass['Class'] =='perl.'),'Class']='u2r'

test_multiclass.loc[(test_multiclass['Class'] =='normal.'),'Class'] = 'normal'


# 
# 

# ### 2.2 Encoding The Dataset:

# In[25]:

# Decoding The Dataset: 
attr_encoder = feature_extraction.DictVectorizer(sparse=False)
label_encoder = preprocessing.LabelEncoder()

train_data_df_m = attr_encoder.fit_transform(train_multiclass.iloc[:,:-1].T.to_dict().values())
train_target_df_m= label_encoder.fit_transform(train_multiclass.iloc[:,-1])


train_data_decoded_m = pd.DataFrame(train_data_df_m)
train_target_decoded_m = pd.DataFrame(train_target_df_m)

test_data_df_m = attr_encoder.transform(test_multiclass.iloc[:,:-1].T.to_dict().values())
test_target_df_m = label_encoder.transform(test_multiclass.iloc[:,-1])

test_data_decoded_m = pd.DataFrame(test_data_df_m)
test_target_decoded_m = pd.DataFrame(test_target_df_m)


print train_data_decoded_m.shape
print test_data_decoded_m.shape


# ### 1.3 Perfroming Feature Reduction using PCA

# In[29]:

#load some modules to help
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


train_data_pca2 = PCA(n_components=29).fit_transform(train_data_decoded_m)
test_data_pca2 = PCA(n_components=29).fit_transform(test_data_decoded_m)

train_data_pca_df2 = pd.DataFrame(train_data_pca2)
test_data_pca_df2 = pd.DataFrame(test_data_pca2)

print train_data_pca_df2.shape
print test_data_pca_df2.shape


# ### 1.4 Normalizing the Data Sets

# In[30]:

#Creating our scaler and applyting it to our dataset after feature reduction
standard_scaler = preprocessing.StandardScaler()
train_ratio_standard_scaled_values2 = standard_scaler.fit_transform(train_data_pca_df2.values)
train_data_scaled2=pd.DataFrame(train_ratio_standard_scaled_values2)

test_ratio_standard_scaled_values2 = standard_scaler.fit_transform(test_data_pca_df2.values)
test_data_scaled2=pd.DataFrame(test_ratio_standard_scaled_values2)


# 
# 

# ## 2 Classification:

# ### 2.1 Using SVM Algorithm

# In[22]:

#Draft
clf = svm.SVC(kernel='linear',class_weight="balanced", max_iter=100000000)
clf.fit(train_data_scaled2, train_target_decoded_m[0])
clf_predict = clf.predict(test_data_scaled2)
print clf.score(test_data_scaled2, test_target_decoded_m)
print metrics.classification_report(test_target_decoded_m, clf_predict)


# ### 2.2 Using Decision Trees Algorithm:

# #### 2.2.1 Performaing Corss Validation on The Training Set for Testing Different Paramter

# In[37]:

## Testing SVM using Different Kernals with class weights balanced
foldnum = 0
fold_results = pd.DataFrame()
criterion=[ 'gini','entropy']
min_samples_leaf = [5,10]
max_depth = [6,12]


for cri in criterion:
    for leaf in min_samples_leaf:
        for depth in max_depth:
            foldnum = 0
            clf3 = tree.DecisionTreeClassifier(criterion=cri,min_samples_leaf=leaf,max_depth=depth,random_state=20160121,class_weight="balanced")
            for train, test in cross_validation.KFold(len(train_data_scaled2), n_folds=5,shuffle=True,random_state=20160202):  
                [ids_tr_data, ids_te_data, ids_tr_target, ids_te_target] = folds_to_split(train_data_scaled_1,train_target_decoded,train, test)
                clf3.fit(ids_tr_data, ids_tr_target[0])
                fold_results.loc[foldnum, 'Accuracy'] = clf3.score(ids_te_data, ids_te_target)
                foldnum+=1 
            print "criterion:",cri
            print "min_samples_leaf:",leaf
            print "max_depth:",depth
            print fold_results.mean()
            print "\n"


# #### 2.2.1 Testing the IDS Model on The Test Set:

# In[46]:

from sklearn.datasets import load_iris
from sklearn import tree

clf3 = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = 5, max_depth=12,random_state=20160121,class_weight="balanced")
clf3.fit(train_data_scaled2, train_target_decoded_m[0])
clf3_predict = clf3.predict(test_data_scaled2)
print "Accuracy :", clf3.score(test_data_scaled2, test_target_decoded_m)
print metrics.classification_report(test_target_decoded_m, clf3_predict)


# ### 2.3 Using Naive Bayes Algorithm:

# In[35]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(train_data_scaled2, train_target_decoded_m[0])
y_pred_predict3 = y_pred.predict(test_data_scaled2)
print y_pred.score(test_data_scaled2, test_target_decoded_m)
print metrics.classification_report(test_target_decoded_m, y_pred_predict3)


# 
# 

# 
# 

# 
# 
# 

# 
# 
# 
# 
# 
# 
# 
# 

# # Building 2 Class IDS Model:

# In[2]:

#let's load the data
train_data_1 = urllib.urlopen('/home/aziz/Downloads/kddcup.data_10_percent_corrected')
test_data_1 = urllib.urlopen('/home/aziz/Downloads/corrected')

#Place both dataset into a dataframe
train_class = pd.read_csv(train_data_1, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])
test_class = pd.read_csv(test_data_1, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])


# 
# 

# ## 1 Pre-Processing The Datasets:

# ### 1.1 Converts Labels to The Right Class

# In[3]:

train_class.loc[(train_class['Class'] !='normal.'),'Class'] = 'attack'
train_class.loc[(train_class['Class'] =='normal.'),'Class'] = 'normal'

test_class.loc[(test_class['Class'] !='normal.'),'Class'] = 'attack'
test_class.loc[(test_class['Class'] =='normal.'),'Class'] = 'normal'


# 
# 

# ### 1.2 Encoding The Dataset

# In[4]:

# Decoding The Dataset: 
attr_encoder = feature_extraction.DictVectorizer(sparse=False)
label_encoder = preprocessing.LabelEncoder()

train_data_df = attr_encoder.fit_transform(train_class.iloc[:,:-1].T.to_dict().values())
train_target_df= label_encoder.fit_transform(train_class.iloc[:,-1])


train_data_decoded = pd.DataFrame(train_data_df)
train_target_decoded = pd.DataFrame(train_target_df)

test_data_df= attr_encoder.transform(test_class.iloc[:,:-1].T.to_dict().values())
test_target_df= label_encoder.transform(test_class.iloc[:,-1])

test_data_decoded = pd.DataFrame(test_data_df)
test_target_decoded = pd.DataFrame(test_target_df)


print train_data_decoded.shape
print test_data_decoded.shape


# 
# 

# ### 1.3 Feature Reduction Using PCA

# In[9]:

#load some modules to help
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


train_data_pca_1 = PCA(n_components=29).fit_transform(train_data_decoded)
test_data_pca_1 = PCA(n_components=29).fit_transform(test_data_decoded)

train_data_pca_df_1 = pd.DataFrame(train_data_pca_1)
test_data_pca_df_1 = pd.DataFrame(test_data_pca_1)

print train_data_pca_df_1.shape
print test_data_pca_df_1.shape


# 
# 

# ### 1.4 Normalizing The Datasets

# In[10]:

#Creating our scaler and applyting it to our dataset after feature reduction
standard_scaler = preprocessing.StandardScaler()
train_ratio_standard_scaled_values = standard_scaler.fit_transform(train_data_pca_df_1.values)
train_data_scaled_1=pd.DataFrame(train_ratio_standard_scaled_values)

test_ratio_standard_scaled_values = standard_scaler.fit_transform(test_data_pca_df_1.values)
test_data_scaled_1=pd.DataFrame(test_ratio_standard_scaled_values)


# 
# 

# ## 2 Classifiying The Data Set

# ### 2.1 Using SVM Algorithm:

# In[8]:

#Draft
lin = svm.SVC(kernel='linear', max_iter=100000000)
lin.fit(train_data_scaled_1, train_target_decoded[0])
lin_predict = lin.predict(test_data_scaled_1)
print lin.score(test_data_scaled_1, test_target_decoded)
print metrics.classification_report(test_target_decoded, lin_predict)
print "Number of support vectors for each class", lin.n_support_
print lin.support_vectors_


# 
# 

# ### 2.2 Using Decision Trees Algorithm

# #### 2.2.1 Performaing Corss Validation on The Training Set for Testing Different Paramter
# 

# In[10]:

## Testing SVM using Different Kernals with class weights balanced
foldnum = 0
fold_results = pd.DataFrame()
criterion=[ 'gini','entropy']
min_samples_leaf = [2, 5, 50]
max_depth = [1,6,12]


for cri in criterion:
    for leaf in min_samples_leaf:
        for depth in max_depth:
            foldnum = 0
            clf = tree.DecisionTreeClassifier(criterion=cri,min_samples_leaf=leaf,max_depth=depth,random_state=20160121,class_weight="balanced")
            for train, test in cross_validation.KFold(len(train_data_scaled_1), n_folds=5,shuffle=True,random_state=20160202):  
                [ids_tr_data, ids_te_data, ids_tr_target, ids_te_target] = folds_to_split(train_data_scaled_1,train_target_decoded,train, test)
                clf.fit(ids_tr_data, ids_tr_target[0])
                clf_predict = clf.predict(ids_te_data)

                fold_results.loc[foldnum, 'Accuracy'] = clf.score(ids_te_data, ids_te_target)
                foldnum+=1 
            print "criterion:",cri
            print "min_samples_leaf:",leaf
            print "max_depth:",depth
            print fold_results.mean()
            print "\n"


# #### 2.2.1 Testing the IDS Model on The Test Set:

# In[11]:


clf_t = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=2,max_depth=12,random_state=20160121,class_weight="balanced")
clf_t.fit(train_data_scaled_1, train_target_decoded[0])
clf_predict = clf_t.predict(test_data_scaled_1)

print "Accuracy (via score):", clf_t.score(test_data_scaled_1, test_target_decoded)
print metrics.classification_report(test_target_decoded, clf_predict)


# ### 2.3 Using Naive Bayes Algorithm:

# In[17]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
Naive = gnb.fit(train_data_scaled_1, train_target_decoded[0])
Naive_predict = Naive.predict(test_data_scaled_1)
print Naive.score(test_data_scaled_1, test_target_decoded)
print metrics.classification_report(test_target_decoded, Naive_predict)
print Naive.class_prior_

