import glob
import os
import numpy as np
from lxml import etree
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

dataColumns = ["headline", "text", "bip:topics", "dc.date.published", "itemID", "XML_File_Name"]
rows = []
paragraph = ""
bipTopicList = []
vec = CountVectorizer(stop_words=None)




def dataExtraction():
    dir = '/Users/amruthkuppili/Downloads/Data/'
    for file in glob.iglob(os.path.join(dir, '*/*.xml')):
        paragraph = ""
        bipTopicCode = ""
        path, fileName = os.path.split(file)  # Obtained File name
        data = etree.parse(file)
        root = data.getroot()
        itemId = data.getroot().attrib.get("itemid")  # Obtained item ID
        headline = data.find("headline").text
        textNode = data.find("text")
        for node in textNode:
            paragraph = paragraph + node.text  # Obtained text
        dcPublishedNode = root.findall("./metadata/dc[@element='dc.date.published']")
        if dcPublishedNode is not None:
            published_date = dcPublishedNode[0].attrib.get("value")  # obtained dc.date.published
        else:
            published_date = "NONE"
        bipNode = root.findall("./metadata/codes[@class='bip:topics:1.0']/code")
        text = removeStopWords(paragraph)  # removing stop words
        if bipNode is not None:
            for innercodes in bipNode:
                bipTopicCode = innercodes.attrib.get("code")  # obtained bip:topic code
                rows.append({"itemID": itemId, "XML_File_Name": fileName, "headline": headline, "text": text,
                             "dc.date.published": published_date, "bip:topics": bipTopicCode})
                uniqueBipTopics(bipTopicCode)
                break
        else:
            bipTopicCode = "NONE"
            rows.append({"itemID": itemId, "XML_File_Name": fileName, "headline": headline, "text": text,
                         "dc.date.published": published_date, "bip:topics": bipTopicCode})

    customDataFrame = pd.DataFrame(rows, columns=dataColumns)
    return customDataFrame



def uniqueBipTopics(topic):
     if topic not in bipTopicList :
        bipTopicList.append(topic)
     return bipTopicList



def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    text_tokens = word_tokenize(text)
    filtered_sentence_list = [w for w in text_tokens if w not in stop_words]
    filtered_lemmatized_list = lemmatization(filtered_sentence_list)
    filtered_stemmed_list = stemming(filtered_lemmatized_list)
    filtered_lemmatized_sentence = ' '.join(filtered_stemmed_list)
    return filtered_lemmatized_sentence



def stemming(sentence):
    ps = PorterStemmer()
    stemmed_words = []
    for w in sentence:
        stemmed_words.append(ps.stem(w))
    return stemmed_words



def lemmatization(filtered_sentence):
    lem = WordNetLemmatizer()
    lemmatized_words = []
    for w in filtered_sentence:
        lemmatized_words.append(lem.lemmatize(w))
    return lemmatized_words




#  Extracting Features from data frame

def featureExtraction(dataFrame):
    textData = dataFrame["text"]
    bipTopicInDataFrame = dataFrame["bip:topics"]
    finalData = textData
    # fileNames = dataFrame["XML_File_Name"]
    featureData = vec.fit_transform(finalData).toarray()
    bipArray = pd.Series(bipTopicInDataFrame).values
    generatedDataFrameData = np.column_stack((featureData,bipArray))
    dataFrameColumns = vec.get_feature_names()
    dataFrameColumns.insert(len(dataFrameColumns),'labels')
    featureDataFrame = pd.DataFrame(data=generatedDataFrameData,columns=dataFrameColumns)
    return featureDataFrame



#  Train/Test Split  division of data into training and testing data
# In this approach we randomly split data into training and testing
# A typical split of training and testing would be around 80% and 20% respectively
# Disadvantages:
#     If the data is not uniformly distributed then there are high chances for bias.
#     In other words if there is huge data and the training data doesn't corresponds to test
#     then there are chances for high bias



def trainTestSplit(dataFrame):
    target = (dataFrame.pop('labels'))
    X_train, X_test, y_train, y_test = train_test_split(dataFrame, target, test_size=0.1, random_state=66)
    return X_train, X_test, y_train, y_test


#  K-Fold division of data into training and testing data
# This approach deals with dividing the whole data into k different subsets
# and later, training is done on k-1 subsets and testing on the last single subset.This process is repeated
# until all the subsets are trained and tested.
# Advantages:
#     AS the training and testing is done all samples it has less bias
#     It is easily understood
#     It is computationally less expensive than the other cross validation approaches such as Leave One Out Cross Validation

def KFoldSplit(dataFrame):
    kf = KFold(n_splits=3, shuffle=True, random_state=4)
    result = next(kf.split(dataFrame), None)
    trainData = dataFrame.iloc[result[0]]
    testData = dataFrame.iloc[result[1]]
    YtrainData = trainData.pop('labels')
    YtestData = testData.pop('labels')
    print(trainData)
    print(YtrainData)
    return trainData, testData, YtrainData, YtestData



def preProcessingSplit(XtrainData, XtestData, YtrainData, YtestData):
    feature_scaler = StandardScaler()
    XtrainData = feature_scaler.fit_transform(XtrainData)
    XtestData = feature_scaler.transform(XtestData)
    pca = PCA(n_components=10)
    XtrainData = pca.fit_transform(XtrainData)
    XtestData = pca.transform(XtestData)

    # lda = LDA(n_components=5)
    # XtrainData = lda.fit_transform(XtrainData, YtrainData)
    # XtestData = lda.transform(XtestData)

    return XtrainData, XtestData, YtrainData, YtestData

def trainingClassifier(Frame, XtrainData, XtestData, YtrainData, YtestData):
    neuralNetwork_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                        hidden_layer_sizes=(60,), random_state=1, max_iter=500)
    trainedClassifier = neuralNetwork_model.fit(XtrainData, YtrainData)
    return trainedClassifier



def classifierDistinction(Frame, XtrainData, XtestData, YtrainData, YtestData ):

    # The main disadvantage of the SVM algorithm is that it has several key parameters that need to be set correctly to achieve the best classification results for any given problem.
    # ANALYSIS:
    # The higher the gamma value the algorithm tries to overfits i.e The higher the gamma value it tries to exactly fit the training data set
    # Increasing C values may lead to overfitting the training data. (C controls the trade off between smooth decision boundary and classifying the training points correctly.)
    SVC_model = SVC(kernel='sigmoid',gamma=0.1,C=0.1)
    SVC_model.fit(XtrainData, YtrainData)
    SVC_prediction = SVC_model.predict(XtestData)


    # Disadvantages are for a Decision tree sometimes calculation can be complex compared to other algorithms.
    # Decision tree often involves higher time to train the model.
    # ANALYSIS:
    # higher the max depth leads to over fitting that is lesser th aucroc and accuracy
    # higher min_samples_split and min_samples_leaf leads to underfitting that is lesser aucroc score
    decionTree_model = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_split=0.2,min_samples_leaf=0.2)
    decionTree_model.fit(XtrainData, YtrainData)
    DT_prediction = decionTree_model.predict(XtestData)


    # Owing to disadvantages, It is complex and requires more computational resources which as a result takes more time
    # This uses lot of memory
    # ANALYSIS
    # n_estimators represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data however, increasing the number of trees decreases the test performance.
    # model overfits for large depth values. The trees perfectly predicts all of the train data, however, it fails to generalize the findings for new data
    # higher min_samples_split and min_samples_leaf leads to underfitting that is lesser aucroc score
    randomForest_model = RandomForestClassifier(n_estimators=10,max_depth=3,min_samples_split=0.4,min_samples_leaf=0.2)
    randomForest_model.fit(XtrainData, YtrainData)
    randomForest_prediction = randomForest_model.predict(XtestData)


    # 1. Easy to implement and can handle redundant data effectively.
    # 2. It has capability to learn non linear models.
    # 3. It can map higher dimensional input to higher dimensional output.
    # 4. It works well with multi-class labels.
    # 5. Talking about disadvantages, It requires tuning of many paramateres like hidden layers,neurons and iterations.
    neuralNetwork_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(60,), random_state=1,max_iter=500)
    neuralNetwork_model.fit(XtrainData, YtrainData)
    neuralNetwork_prediction = neuralNetwork_model.predict(XtestData)


    # 1. The linear regression technique involves the "continuous dependent variable" and the independent variables can be continuous or discrete.
    # 2. It's disadvantage is that if we have a number of parameters than the number of samples available then the model starts to model the noise rather than the relationship between the variables.
    # 3. Eventhough it is simple and has scientific acceptance it is sensitive to outliers
    # 4. It can easily overfit
    linearRegression_model = LinearRegression()
    linearRegression_model.fit(XtrainData, YtrainData)
    linearRegression_prediction =  linearRegression_model.predict(XtestData)




def calculateMetrics(trainedClassifier,YtestData):

     # Confusion matrix is used to find the correctness and accuracy of the model
     # It is used for Classification problem where the output can be of two or more types of classes.
     predictor = trainedClassifier.predict(XtestData)
     confusionMatrix = confusion_matrix(predictor, YtestData)

     # Accuracy in classification problems is the number of correct predictions made by the model over all kinds predictions made.
     # In the Numerator, are our correct predictions (True positives and True Negatives)and in the denominator, are the kind of all predictions made by the algorithm.
     accuracy = accuracy_score(predictor, YtestData)

     # ROC is a probability curve and AUC represents degree or measure of separability
     #It tells how much model is capable of distinguishing between classes.
     # For multi-class labels LabelBinarizer can be used to evaluate AUC ROC score
     # LabelBinarizer converts multiple lables into binary lables with a transform method
     lb = LabelBinarizer()
     lb.fit(YtestData)
     y_test = lb.transform(YtestData)
     y_pred = lb.transform(predictor)
     rocaucScore = roc_auc_score(y_test, y_pred, average="macro")   #AUC-ROC score


     # classification report contains Precision, Recall, f1-score and support
     # Precision: Precision is the proportion of relevant results
     # Recall: percentage of total relevant results correctly classified by the algorithm.
     # f1-score: This is the combination of both Precision and Recall (Harmonic mean of the both)
     # Support: The number of true instances for each label
     ClassificationReport = classification_report(predictor, YtestData)
     print("Accuracy : ",accuracy)
     print("ROC_AUC Score :", rocaucScore)
     print("Classification Report :")
     print(ClassificationReport)


customDataFrame = dataExtraction()
featureDataFrame = featureExtraction(customDataFrame)
totalDataFrame = featureDataFrame[:]
#featureDataFrame.to_csv('/Users/amruthkuppili/Downloads/samplecsv/featuredata.csv')
XtrainData, XtestData, YtrainData, YtestData = trainTestSplit(totalDataFrame)
#XtrainData, XtestData, YtrainData, YtestData = KFoldSplit(featureDataFrame)
XtrainData, XtestData, YtrainData, YtestData = preProcessingSplit(XtrainData, XtestData, YtrainData, YtestData)
receiveTrainedClassifier = trainingClassifier(featureDataFrame, XtrainData, XtestData, YtrainData, YtestData)
calculateMetrics(receiveTrainedClassifier, YtestData)
classifierDistinction(featureDataFrame, XtrainData, XtestData, YtrainData, YtestData)
