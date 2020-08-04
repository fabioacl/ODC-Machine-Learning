# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:42:47 2019

@author: fabio
"""

from keras import regularizers
from keras.models import Sequential,Model
from keras.layers import Dense,Input,TimeDistributed,Bidirectional,Add
from keras.layers import Activation
from keras.layers import LSTM,Conv1D,MaxPooling1D,Flatten
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")
import itertools
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.feature_extraction.text import *
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from scipy import sparse
import pickle
import gensim
import nltk
import numpy as np
from pathlib import Path
import pandas as pd
from keras import backend as K

##Functions

try:
	model
except:
	print('Loading Model...')
	model=gensim.models.Word2Vec.load("odcembeddings.model")
	print('Model Loaded!')
	
def tokenization(X_data):
	X_dataset_tokens=[]
	for document in X_data:
		X_dataset_tokens.append(nltk.word_tokenize(document))
	return X_dataset_tokens
def padding(X_dataset,maximum_len):
	for doc_idx in range(len(X_dataset)):
		while len(X_dataset[doc_idx])<maximum_len:
			X_dataset[doc_idx].append(0,'padding')
	for doc_idx in range(len(X_dataset)):
		while len(X_dataset[doc_idx])>maximum_len:
			del X_dataset[doc_idx][-1]
	return X_dataset
def doc_to_embedding(X_data,classifier='deep'):
	k=0
	X_dataset=[]
	for document in X_data:
		X_dataset.append([])
		for word in document:
			try:
				if word=='padding':
					X_dataset[k].append(np.zeros(100))
				else:
					X_dataset[k].append(np.array(model.wv[str(word)]))
			except:
				X_dataset[k].append(np.array(model.wv['unknown']))
		X_dataset[k]=np.array(X_dataset[k])
		k=k+1
	return X_dataset

class DeepLearningModels():
	def __init__(self,architecture,lstm_units=64,filters=32,kernel_size=5,pool_size=2,dense_units=16,dropout=0.4,epochs=15):
		self.architecture=architecture
		self.lstm_units=lstm_units
		self.dropout=dropout
		self.filters=filters
		self.kernel_size=kernel_size
		self.pool_size=pool_size
		self.dense_units=dense_units
		self.epochs=epochs
		
	def fit(self,X_train,y_train):
		X_train_tokens=tokenization(X_train)
		self.document_lengths=[len(doc) for doc in X_train_tokens]
		X_train_pad=padding(X_train_tokens,np.mean(self.document_lengths)+3*np.std(self.document_lengths))
		X_train=np.array(doc_to_embedding(X_train_pad))
		num_classes=max(y_train)
		y_train=np.array([to_categorical(i,num_classes=num_classes+1) for i in y_train])
			
		if self.architecture=='CNN':
			sequence_input = Input(shape=(X_train[0].shape[0],X_train[0].shape[1],))
			l_cov1= Conv1D(self.filters, self.kernel_size, activation='relu')(sequence_input)
			l_pool1 = MaxPooling1D(self.pool_size)(l_cov1)
			l_dropout1=Dropout(self.dropout)(l_pool1)
			l_flat = Flatten()(l_dropout1)
			l_dense = Dense(self.dense_units, activation='relu')(l_flat)
			preds = Dense(num_classes+1, activation='softmax')(l_dense)
			model = Model(sequence_input, preds)
			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
		elif self.architecture=='RNN':
			sequence_input = Input(shape=(X_train[0].shape[0],X_train[0].shape[1],))
			l_lstm = Bidirectional(LSTM(self.lstm_units))(sequence_input)
			preds = Dense(max(all_bug_classes)+1, activation='softmax')(l_lstm)
			model = Model(sequence_input, preds)
			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
		model.fit(X_train, y_train,epochs=self.epochs,batch_size=32,verbose=2)
		self.model=model
		
	def predict(self,X):
		predict_classes=[]
		real_classes=[]
		X=tokenization(X)
		X=padding(X,np.mean(self.document_lengths)+3*np.std(self.document_lengths))
		X=np.array(doc_to_embedding(X))
		y_pred_prob=self.model.predict(X)
		#
		for prob in y_pred_prob:
			prediction=np.where(prob==max(prob))
			predict_classes.append(prediction[0][0])
		return predict_classes
	 

def savedata(cm_matrixes,results,category,classifier):
	writer = pd.ExcelWriter(category+'_results_'+classifier+'.xlsx')
	cm_matrixes.to_excel(writer,'Confusion Matrixes')
	results.to_excel(writer,'Results')
	writer.save()
def savedata_run(results,classifier,category):
	writer = pd.ExcelWriter(category+'_dataset_'+classifier+'.xlsx')
	results.to_excel(writer,'Results')
	writer.save()
def plot_confusion_matrix(cm, classes,category,
						  normalize=False,
						  title='Impact',
						  cmap=plt.cm.summer):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	It was imported from scikit-learn documentation.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)
	
	f=plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else '.2f'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="black" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	f.savefig("ConfusionMatrix"+category+".pdf", bbox_inches='tight')

def mean_accuracies(all_accuracies,y):
	number_classes=len(np.unique(y))
	accuracies=[0]*number_classes
	for i in range(len(all_accuracies)):
		for j in range(len(accuracies)):
			accuracies[j]+=all_accuracies[i][j]
	for i in range(len(accuracies)):
		accuracies[i]=accuracies[i]/len(all_accuracies)
	return accuracies

def str_to_int(array):
	output_array=[]
	for i in array:
		output_array.append(int(i.strip('\n')))
	return output_array

def file_to_array(filename):
	array=[]
	filepath=Path(filename)
	if filepath.is_file():
		f=open(filename,'r')
		array=f.readlines()
		f.close()
	return array

def choose_random_data(all_data,balance,samples_percent,dataset_percent=1.0):
	labels_size=[]
	all_samples=[]
	all_classes=[]
	used_labels=[]
	
	for label in all_data:
		labels_size.append(len(all_data[label]))
	minimum_samples=round(max(labels_size)*samples_percent)
	filtered_sizes=[]
	for size in labels_size:
		if size>=minimum_samples:
			filtered_sizes.append(size)
	minimum_samples=sorted(filtered_sizes)[0]
	k=0 #Class number
	for label in all_data:
		label_texts=all_data[label]
		if len(label_texts)>=minimum_samples:
			if balance==True:
				min_samples=round(minimum_samples*dataset_percent)
				index_array=random.sample(range(0, len(label_texts)), min_samples)
				label_texts=[label_texts[i] for i in index_array]
				all_classes.extend([k]*min_samples)
			else:
				all_classes.extend([k]*len(label_texts))
				min_samples=minimum_samples
			all_samples.extend(label_texts)
			used_labels.append(label)
			k=k+1
	return all_samples,all_classes,used_labels,labels_size,min_samples

def retrieve_ids(database,labels):
	database_ids={}
	for index in range(len(labels)):
		database_ids[labels[index]]=str_to_int(file_to_array(labels[index]+" "+database+".txt"))
	return database_ids

def retrieve_data(all_db_ids):
	all_bug_data={}
	for db in all_db_ids:
		for label in all_db_ids[db]:
			if label not in all_bug_data:
				all_bug_data[label]=[]
			for bug_id in all_db_ids[db][label]:
				string=""
				filename=db+"/"+db+" - "+str(bug_id)+".txt"
				filepath=Path(filename)
				if filepath.is_file():
					f=open(filename,'r')
					content=f.read()
					string=content
					if string!="":
						all_bug_data[label].append(string)
					else:
						print(bug_id)
					f.close()
	return all_bug_data,string

def filter_words(text,category):
	notvaluable_chars=['.',',','(',')','[',']','-','_',':',';','\\','/','"',"'",'<','>','*','+','#','?','!',
					   '«','»','=','@','$','%','&','{','}']
	filtered_text=''
	for char in text:
		if char.isdigit()==False and char not in notvaluable_chars:
			filtered_text+=char
		else:
			filtered_text+=' '
	text_split=filtered_text.split()
	notvaluable_words=['HBASE','CASSANDRA','SERVER']
	new_text_split=[]
	for word in text_split:
		if word not in notvaluable_words:
			new_text_split.append(word)
	erase_words=True
	new_text=""
	last_word=""            
	for word in new_text_split:
		if word=='Link' or word=='Dates' or word=='Comments' or word=='Comment' or word=='Author':
			erase_words=True
		if erase_words==False:
			lowercase_first=lambda s:s[:1].lower() + s[1:] if s else ''
			upper_chars=0
			word=str(word)
			for char in word:
				if char.isupper():
					upper_chars+=1
			if upper_chars==1:
				word=lowercase_first(word)
			new_text=new_text+word+" "
		if category=='Activity' or category=='Code_Inspection' or category=='Function_Test' or category=='System_Test' or category=='Unit_Test' or category=='Impact':
			if word=='Title' or word=='Description':
				erase_words=False
		else:
			if word=='Title' or word=='Description' or (last_word=='Comment' and word[0]=='#') or word=='Message':
				erase_words=False
		last_word=word
	return new_text

def filter_documents(X_train,Y_train,words_class):
	new_X_train=[]
	for doc_idx in range(len(X_train)):
		new_doc=""
		doc_split=X_train[doc_idx].split()
		for w in doc_split:
			if w in words_class[Y_train[doc_idx]]:
				new_doc=new_doc+w+" "
		new_X_train.append(new_doc)
	return new_X_train

database_labels=["MongoDB","Cassandra","HBase"]
activity_labels=["Code Inspection","Function Test","System Test","Unit Test","Design Review"]
code_inspection_labels=["Backward Compatibility","Concurrency","Design Conformance","Internal Document",
						"Language Dependency","Lateral Compatibility","Logic_Flow","Rare Situation",
						"Side Effects"]
function_test_labels=["Test Coverage","Test Sequencing","Test Variation","Test Interaction"]
system_test_labels=["Blocked Test","Recovery_Exception","Software Configuration","Startup_Restart",
					"Workload_Stress","Hardware Configuration"]
unit_test_labels=["Complex Path","Simple Path"]
impact_labels=["Capability","Installability","Integrity_Security","Interaction",
			   "Migration","Performance","Reliability","Requirements",
			   "Standards","Serviceability","Usability","Maintenance","Documentation",
			   "Accessibility"]
target_labels=["Requirements_Target","Design","Code","Build_Package",
			   "Information Development","National Language Support"]
defect_labels=["Algorithm_Method","Assignment_Initialization","Checking","FCO",
			   "Interface_OOMessages","Timing_Serialization","Relationship"]
qualifier_labels=["Missing","Incorrect","Extraneous"]
		
recall_Classifier=[]
precision_Classifier=[]
accuracy_Classifier=[]
D=[1,2,3,4,5,6,7,8,9,10,15]
df=2
runs=25

print("1:Activity\n2:Code Inspection\n3:Function Test\n4:System Test\n5:Unit Test\n6:Impact"\
	  "\n7:Target\n8:Defect Type\n9:Qualifier")
chosen_attribute=int(input("Introduza a opção que pretende: "))
print("1:Balancear dados\n2:Não balancear dados")
chosen_balanced=int(input("Introduza a opção que pretende: "))
print("1:Parameters GridSearch\n2:Dataset Percentage\n3:Final Results")
chosen_output=int(input("Introduza a opção que pretende: "))
if chosen_balanced==1:
	balanced_data=True
else:
	balanced_data=False

if chosen_attribute==1:
	chosen_labels=activity_labels
	category='Activity'
	samples_percent=0.05
	c=1
	k=17
	nr_trees=1024
elif chosen_attribute==2:
	chosen_labels=code_inspection_labels
	category='Code_Inspection'
	samples_percent=0.05
	k=17
	nr_trees=512
	c=2
elif chosen_attribute==3:
	chosen_labels=function_test_labels
	category='Function_Test'
	samples_percent=0.05
	k=5
	nr_trees=512
	c=2
elif chosen_attribute==4:
	chosen_labels=system_test_labels
	category='System_Test'
	samples_percent=0.05
	k=11
	nr_trees=256
	c=1
elif chosen_attribute==5:
	chosen_labels=unit_test_labels
	category='Unit_Test'
	samples_percent=0.05
	k=19
	nr_trees=128
	c=0.25
elif chosen_attribute==6:
	chosen_labels=impact_labels
	category='Impact'
	samples_percent=0.02
	k=21
	nr_trees=512
	c=1
elif chosen_attribute==7:
	chosen_labels=target_labels
	category='Target'
	samples_percent=0.05
	k=15
	nr_trees=512
	c=16
elif chosen_attribute==8:
	chosen_labels=defect_labels
	category='Defect_Type'
	samples_percent=0.05
	nr_trees=256
	c=8
	k=7
else:
	chosen_labels=qualifier_labels
	category='Qualifier'
	samples_percent=0.05
	k=17
	nr_trees=256
	c=16

mean_accuracy_parameter=[]
std_accuracy_parameter=[]
mean_recall_parameter=[]
std_recall_parameter=[]
mean_precision_parameter=[]
std_precision_parameter=[]
C=[2**-5,2**-4,2**-3,2**-2,2**-1,1,2,4,8,16,32]
Polynomial_Degrees=[1,2,3,4,5]
Gammas=[2**-5,2**-4,2**-3,2**-2,2**-1,1,2,4,8,16,32]
K_values=[1,3,5,7,9,11,13,15,17,19,21]
Number_trees=[1,2,4,8,16,32,64,128,256,512,1024]
LSTM_units=[2**i for i in range(3,8)]
dropout_percentages=[0.1*i for i in range(1,6)]

dataset_percent=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
choose_classifier=int(input('1:SVM\n2:KNN\n3:Random Forest\n4:Naive Bayes\n5:Nearest Centroid\n6:Voting\n7:Poly Non-Linear SVM\n8:RBF Non-Linear SVM\n9:Deep Learning\nEscolha o classificador:'))
if choose_classifier==1:
	classifier='SVM'
elif choose_classifier==2:
	classifier='KNN'
elif choose_classifier==3:
	classifier='RF'
elif choose_classifier==4:
	classifier='NB'
elif choose_classifier==5:
	classifier='NC'
elif choose_classifier==6:
	classifier='Voting'
elif choose_classifier==7:
	classifier='Poly Non-Linear SVM'
elif choose_classifier==8:
	classifier='RBF Non-Linear SVM'
elif choose_classifier==9:
	choose_architecture=int(input('1:Convolutional Neural Network\n2:Recurrent Neural Network\nEscolha a arquitetura:'))
	if choose_architecture==1:
		classifier='CNN'
	else:
		classifier='RNN'
print('Classificador:',classifier)

percent=1
parameters=dataset_percent
parameters_save=[]

cm_matrixes=pd.DataFrame()
for b in range(runs):
	print("Round: "+str(b))
	
	#Load Dataset
	mdb_ids=retrieve_ids(database_labels[0],chosen_labels)
	cas_ids=retrieve_ids(database_labels[1],chosen_labels)
	hb_ids=retrieve_ids(database_labels[2],chosen_labels)
	all_database_ids={"MongoDB":mdb_ids,"Cassandra":cas_ids,"HBase":hb_ids}
	all_bug_data,string=retrieve_data(all_database_ids)
	
	#Dataset Balancing
	all_bug_data_balanced,all_bug_classes,used_labels,labels_size,min_samples=choose_random_data(all_bug_data,balanced_data,samples_percent,percent)
	for doc_index in range(len(all_bug_data_balanced)):
		all_bug_data_balanced[doc_index]=filter_words(all_bug_data_balanced[doc_index],category)
	
	if b==0: #First run
		cm=np.zeros(shape=(len(used_labels),len(used_labels)))
	classifiers_Classifier=[]
	acc_Classifier=[]
	real_Classifier=[]
	predicted_Classifier=[]
	all_bug_data_balanced=np.array(all_bug_data_balanced)
	
	if min_samples<10:
		k_fold=min_samples
	else:
		k_fold=10
	loo=StratifiedKFold(n_splits=k_fold,shuffle=False)
	#GRID SEARCH
	C=[2**-5,2**-4,2**-3,2**-2,2**-1,1,2,4,8,16,32]
	Polynomial_Degrees=[1,2,3,4,5]
	Gammas=[2**-5,2**-4,2**-3,2**-2,2**-1,1,2,4,8,16,32]
	K_values=[1,3,5,7,9,11,13,15,17,19,21]
	Number_trees=[1,2,4,8,16,32,64,128,256,512,1024]
	# for c in C:   # SVM
	# 	for degree in Polynomial_Degrees:   # Polynomial SVM
	# 	for gamma_value in Gammas:   # RBF SVM
	# for k in K_values:   # KNN
	# for nr_trees in Number_trees:   # Random Forest
	for LSTM_unit in LSTM_units:   # LSTM-based Network
		for dropout_percent in dropout_percentages:   # LSTM-based Network
			recall_Classifier=[]
			precision_Classifier=[]
			accuracy_Classifier=[]
			class_accuracy_Classifier=[]
			if classifier=='SVM':
				print("C="+str(c))
			elif classifier=='RF':
				print("Trees="+str(nr_trees))
			elif classifier=='KNN':
				print("K="+str(k))
			elif classifier=='Voting':
				print("C="+str(c))
				print("Trees="+str(nr_trees))
				print("K="+str(k))
			elif classifier=='Poly Non-Linear SVM':
				print("C="+str(c))
				print("Degrees="+str(degree))
				parameters=(c,degree)
			elif classifier=='RBF Non-Linear SVM':
				print("C="+str(c))
				print("Gamma="+str(gamma_value))
				parameters=(c,gamma_value)
			elif classifier=='RNN':
				print('LSTM units=',LSTM_unit)
				print('Dropout Percent=',dropout_percent)
			for train_index, test_index in loo.split(all_bug_data_balanced,all_bug_classes):
				#Split Dataset
				X_train_raw, X_test_raw = all_bug_data_balanced[train_index], all_bug_data_balanced[test_index]
				Y_train, Y_test = [all_bug_classes[i] for i in train_index], [all_bug_classes[j] for j in test_index]
				
				#Feature Extraction
				if classifier in ['RNN','CNN']:
					X_train_tokens=tokenization(X_train_raw)
					document_lengths=[len(doc) for doc in X_train_tokens]
					X_train_pad=padding(X_train_tokens,np.mean(document_lengths))
					X_train=np.array(doc_to_embedding(X_train_pad,'deep'))
					X_test_tokens=tokenization(X_test_raw)
					X_test_pad=padding(X_test_tokens,np.mean(document_lengths))
					X_test=np.array(doc_to_embedding(X_test_pad,'deep'))
				else:
					count_vect = TfidfVectorizer(min_df=df,ngram_range=(1,3))
					X_train = count_vect.fit_transform(X_train_raw).toarray()
					X_test=count_vect.transform(X_test_raw).toarray()
					#Dimensionality Reduction
					pca=PCA(n_components=round(len(X_train)*0.5))
					X_train=pca.fit_transform(X_train)
					X_test=pca.transform(X_test)
				
				#Create weight vector
				if balanced_data==False:
					classes_index=np.unique(Y_train)
					classes_size=[]
					initial_weights_dict={}
					for i in range(len(classes_index)):
						classes_size.append(np.count_nonzero(np.array(Y_train)==i))
					total_size=len(Y_train)
					for i in range(len(classes_index)):
						initial_weights_dict[classes_index[i]]=(total_size-classes_size[i])/total_size
				else:
					initial_weights_dict='balanced'
				
				#Fit classifiers
				if 'SVM' in classifier:
					if classifier=='SVM':
						clf_text=SVC(kernel='linear',class_weight=initial_weights_dict,C=c,probability=True)
					elif classifier=='RBF Non-Linear SVM':
						clf_text=SVC(kernel='rbf',class_weight=initial_weights_dict,C=c,gamma=gamma_value,probability=True)
					else:
						clf_text=SVC(kernel='poly',class_weight=initial_weights_dict,C=c,degree=degree,probability=True)
				elif classifier=='KNN':
					clf_text=KNeighborsClassifier(n_neighbors=k,metric='cosine')
				elif classifier=='NB':
					clf_text=GaussianNB(priors=initial_weights_dict)
				elif classifier=='RF':
					clf_text=RandomForestClassifier(n_estimators=nr_trees,class_weight=initial_weights_dict)   
				elif classifier=='NC':
					clf_text=NearestCentroid(metric='cosine')
				elif classifier=='Voting':
					clf_text1=SVC(kernel='linear',class_weight=initial_weights_dict,C=c,probability=True)
					clf_text2=NearestCentroid(metric='cosine')
					clf_text3=KNeighborsClassifier(n_neighbors=k,metric='cosine')
					clf_text4=RandomForestClassifier(n_estimators=nr_trees,class_weight=initial_weights_dict) 
					clf_text5=GaussianNB(priors=None)
					clf_text=VotingClassifier(estimators=[
							('svm', clf_text1), ('nc', clf_text2), ('knn', clf_text3)], voting='soft')
				elif classifier in ['RNN','CNN']:
					real_classes=[]
					Y_test=np.array([to_categorical(i,num_classes=max(all_bug_classes)+1) for i in Y_test])
					for prob in Y_test:
						prediction=np.where(prob==max(prob))
						real_classes.append(prediction[0][0])
					Y_test=real_classes
					clf_text=DeepLearningModels(classifier,lstm_units=LSTM_unit,filters=32,kernel_size=5,
												pool_size=2,dense_units=16,dropout=dropout_percent,epochs=15)
					
				if classifier in ['RNN','CNN']:
					clf_text.fit(X_train_raw,Y_train)
				else:
					clf_text.fit(X_train,Y_train)
				
				#Predict Classes and Check Results
				if classifier in ['RNN','CNN']:
					predict_classes=clf_text.predict(X_test_raw)
				else:
					predict_classes=clf_text.predict(X_test)
				K.clear_session()
				predicted_Classifier=predicted_Classifier+list(predict_classes)
				real_Classifier=real_Classifier+list(Y_test)
				report_Classifier=classification_report(list(Y_test),list(predict_classes),output_dict=True)
				recall_Classifier.append(float(report_Classifier['weighted avg']['recall']))
				precision_Classifier.append(float(report_Classifier['weighted avg']['precision']))
				accuracy_Classifier.append(accuracy_score(Y_test,predict_classes))
				
			cmiter=confusion_matrix(real_Classifier,predicted_Classifier)
			cm_matrixes=cm_matrixes.append(pd.DataFrame(data=cmiter),ignore_index=True)
			cm_matrixes=cm_matrixes.append([["" for i in range(len(used_labels))]],ignore_index=True)
			cm=cm+cmiter
			print("Accuracy "+classifier+": "+str(np.mean(accuracy_Classifier))+"+-"+str(np.std(accuracy_Classifier)))
			print("Recall "+classifier+": "+str(np.mean(recall_Classifier))+"+-"+str(np.std(recall_Classifier)))
			print("Precision "+classifier+": "+str(np.mean(precision_Classifier))+"+-"+str(np.std(precision_Classifier)))

			if chosen_output==1 or chosen_output==2:
				mean_accuracy_parameter.append(np.mean(accuracy_Classifier))
				std_accuracy_parameter.append(np.std(accuracy_Classifier))
				mean_recall_parameter.append(np.mean(recall_Classifier))
				std_recall_parameter.append(np.std(recall_Classifier))
				mean_precision_parameter.append(np.mean(precision_Classifier))
				std_precision_parameter.append(np.std(precision_Classifier))
				parameters_save.append(str((LSTM_unit,dropout_percent)))
				results=pd.DataFrame(data=[mean_accuracy_parameter,std_accuracy_parameter,mean_recall_parameter,
										   std_recall_parameter,mean_precision_parameter,std_precision_parameter],
					index=['Accuracy','Accuracy_Std','Recall','Recall_Std','Precision','Precision_Std'],columns=parameters_save)
				savedata_run(results,classifier,category)
