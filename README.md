# covid19prediction
it is basically a machine learning model which predicts wether you have covid or not on basis of symtoms u are experiencing.

	** Problem statement HACKATHON	** 
healthcare-A predictive analysis of epidemic and pandemic outbreaks in advance using cost effective and sustainable digital mechanisms.
I created xg boost model for covid prediction which will help us to contorl the pandemic and epidemic outbreaks by examining 20 symptoms ehich will give prediction wether person has covid or not.

	**  XG BOOST 	** 
what is xg boost?

XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. However, when it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now. 
THE PROCEDURE FOR XGBOOST ALGORITHM IS-
1 Decision Tree: Every decision tree has a set of criteria which are features and parameters like for example  for teacher hiring criteria is education level, number of years of experience, interview performance. A decision tree is analogous to a teacher hiring manger interviewing candidates based on his or her own criteria.
2 Bagging: Now imagine instead of a single teacher hiring interviewer , now there is an whole panel where each interviewer has a vote. Bagging or bootstrap aggregating involves combining inputs from all interviewers for the final decision through a democratic voting process.
3 Random Forest: It is a bagging-based algorithm with a key difference wherein only a subset of features is selected at random. In other words, every interviewer will only test the interviewee on certain randomly selected qualifications (e.g. a english interview for testing english speaking skills and a behavioral interview for evaluating non-technical skills).
4 Boosting: This is an alternative approach where each interviewer alters the evaluation criteria based on feedback from the previous interviewer. This ‘boosts’ the efficiency of the interview process by deploying a more dynamic evaluation process.
5 Gradient Boosting: A special case of boosting where errors are minimized by gradient descent algorithm e.g. the aptitude skills firms will leverage by using case interviews to filter out less qualified candidates.
6 XGBoost: Think of XGBoost as gradient boosting on ‘steroids’ (thats why it is called extreme gradient boosting). It is a perfect combination of software and hardware optimization techniques to yield superior results using less computing resources in the shortest amount of time.


Why does XGBoost perform so well?
 However, XGBoost improves upon the base GBM framework through systems optimization and algorithmic enhancements.
The library provides a system for use in a range of computing environments, not least:
Parallelization of tree construction using all of your CPU cores during training.
Distributed Computing for training very large models using a cluster of machines.
Out-of-Core Computing for very large datasets that don’t fit into memory.
Cache Optimization of data structures and algorithm to make best use of hardware.
System Optimization:
1 Parallelization: XGBoost approaches the process of sequential tree building using parallelized implementation. This is possible due to the interchangeable nature of loops used for building base learners; the outer loop that enumerates the leaf nodes of a tree, and the second inner loop that calculates the features. This nesting of loops limits parallelization because without completing the inner loop (more computationally demanding of the two), the outer loop cannot be started. Therefore, to improve run time, the order of loops is interchanged using initialization through a global scan of all instances and sorting using parallel threads. This switch improves algorithmic performance by offsetting any parallelization overheads in computation.
2 Tree Pruning: The stopping criterion for tree splitting within GBM framework is greedy in nature and depends on the negative loss criterion at the point of split. XGBoost uses ‘max_depth’ parameter as specified instead of criterion first, and starts pruning trees backward. This ‘depth-first’ approach improves computational performance significantly.
3 Hardware Optimization: This algorithm has been designed to make efficient use of hardware resources. This is accomplished by cache awareness by allocating internal buffers in each thread to store gradient statistics. Further enhancements such as ‘out-of-core’ computing optimize available disk space while handling big data-frames that do not fit into memory.

	**  K fold cross validation   	**
  it is applied to maximize the parametrs and to test on various test which help us to determine the best performance for our model.

	**  DATASET 	** 
  IT CONSITIS OF 20 FEATURES (SYMPOTOMS) , AND DEPENDENT VARIBLE/OUTPUT i.e. WETHER PERSON HAS COVID OR NOT.
  The dataset had imbalanced distribution but i cleaned the dataset and made it balanced.
  PREDICTION-98.21%
  Avg Bias: 0.012
  Avg Variance: 0.008
     OUTPUT SCREENSHOT-
   ![Screenshot (474)](https://user-images.githubusercontent.com/90260133/144844055-1fdcfcf4-6581-4880-ad05-564e56088871.png)
  Accuracy: 98.73 %
  Standard Deviation: 1.18 %
  ![Screenshot (476)](https://user-images.githubusercontent.com/90260133/144844264-190bb319-af54-468e-9b12-b744e4291036.png)

    
    
  
