%                                             -*- coding: utf-8 -*-
% Using LaTeX is highly recommended. 
% If you correct/upgrade anything please send it back to help others' work.

\documentclass[a4paper,oneside]{article}
\usepackage[margin=3cm]{geometry}
% =================================================================
\usepackage[english]{babel}
\selectlanguage{english}

%=================================================================
% Font encoding
% The T1 font encoding is an 8-bit encoding and uses fonts that have 256 glyphs.
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{multirow} 
%================================================================
% Use Times New Roman
\usepackage{times}

%================================================================
% Figures
% usage: \includegraphics[width=<width>]{fig.png}
\usepackage{graphicx}

% images root folder
\graphicspath{{./figs/}}
\usepackage{placeins}
\usepackage{caption}
\usepackage{tabularx} % For automatic line breaking in tables
\usepackage{booktabs} % For better horizontal lines
\usepackage{array} % for the m{} column type

%================================================================
% Package to create pdf hyperlinks
%------------------------------------
% Hyperref should be the last imported (except some problematic packages, e.g. algorithm)
\usepackage[colorlinks=true]{hyperref}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HERE IS THE START OF THE DOCUMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{document}
\input{macros} % Import macros
\markright{Dávid Nagy (A936R6)} % one sided title page!!!
%--------------------------------------------------------------------
% title page
%--------------------------------------------------------------------

\begin{titlepage}
%bme logo 
 \begin{figure}[h]
    \centering
      \includegraphics[width=12cm]{bme_logo.pdf}
  \label{fig:bme_logo}
  \end{figure}
  \thispagestyle{empty}
  
  % generate title
  \projectlaboratorytitle
 
  \projectlaboratoryauthor{Dávid Nagy}{A936R6}{Infocommunication}{nagydavid02@gmail.com}{Bálint Gyires-Tóth, PhD}{toth.b@tmit.bme.hu}{Dániel Unyi}{unyi.daniel@tmit.bme.hu}
 
 
  %\tasktitle
  \tasktitle{Classification of Gene Expression Data Using Deep Learning} 

  %\taskdescription
  \taskdescription{The goal is to build a machine learning model that can classify patients with different types of cancer, based on their gene expression data. First, I want to explore whether there is a machine learning model that can learn on the cBioPortal dataset \cite{cbio}, second I want to achieve an accuracy as high as possible.}

  % semester Arguments: #1=Semester (format: xxxx/xx , #2=I/II (without dot!))
  \begin{center}
    \semester{2023/24}{II}
  \end{center}
  
\end{titlepage} 

%==================================================================
% ------------ CONTENT ------------
\section{Theory and previous works}
\label{sec:theory_prev}

\subsection{Introduction}
\label{sec:introduction}

Artificial intelligence (AI) is a rapidly advancing field of computer science focused on creating systems that can perform tasks that typically require human intelligence. These tasks include understanding natural language, recognizing patterns, learning from experience, and making decisions. AI has applications in various domains, from self-driving cars to personalized recommendations on streaming platforms, and its development holds the potential to profoundly impact industries and societies worldwide. To get an overview, as well as a deeper understanding of AI, Stuart Russel, and Peter Norvig's book is recommended \cite{aibook}.

Machine learning (ML) is a branch of AI that enables systems to learn from data and improve their performance over time without being explicitly programmed. It encompasses a variety of techniques that allow computers to recognize patterns, make predictions, and learn from experience. Bishop's book is recommended for further reading in ML \cite{bishop_book}.

Deep learning (DL) is a subset of ML that deals with algorithms inspired by the structure and function of the human brain, known as artificial neural networks. These networks consist of multiple layers of interconnected nodes (neurons) that process information in a hierarchical manner. DL has revolutionized fields like computer vision, natural language processing, and speech recognition, achieving remarkable performance in tasks that were once considered very challenging \cite{dl}. Goodfellow, Bengio, and Courville's book is a good starting point for delving deeper into this field \cite{dlbook}.

The connection between gene expression data and ML is significant in the field of bioinformatics and computational biology \cite{dl_bio}. Gene expression data provides information about the activity of genes in cells, tissues, or organisms under different conditions.

ML techniques, particularly DL algorithms, have been increasingly applied to analyze gene expression data. These methods can identify patterns, correlations, and predictive models from large-scale datasets, helping to uncover hidden relationships between genes, identify disease biomarkers, predict clinical outcomes, and even suggest potential drug targets \cite{dl_gene}.

The utilization of ML in analyzing gene expression data offers a powerful tool for understanding the complex mechanisms underlying biological processes and diseases. It enables researchers to extract meaningful information from multi-dimensional datasets (like the two-dimensional datasets in \cite{cbio}), paving the way for more targeted and personalized approaches in medicine and biotechnology.

In this paper, I present a way (in section \ref{subsection:finish_line}) to extract information from the given dataset (\cite{cbio}) by classifying cancer patients based on their gene expression data. I also point out future directions for this study to continue and expand the capabilities of the model I built.

\subsection{Theoretical summary}
\label{sec:theoretical_summary}

\subsubsection{Supervised, unsupervised and reinforcement learning}

The three main branches of ML are supervised learning, unsupervised learning, and reinforcement learning.

In supervised learning, the model is trained on a labeled dataset, where each input is associated with a corresponding target or output. The goal is to learn a mapping (the model has to learn a function) from inputs to outputs, allowing the model to make predictions or decisions when given new, unseen data. Common tasks include classification and regression (see section \ref{subsec:class_reg}).

In unsupervised learning, the model is given an unlabeled dataset and must find patterns or structure in the data without explicit guidance. Unlike supervised learning, there are no predefined target variables. Unsupervised learning algorithms aim to discover hidden patterns, group similar data points, or reduce the dimensionality of the data. An example could be anomaly detection.

Reinforcement learning involves training an agent (``an agent is nothing more than something that acts'' - as stated in \cite{aibook}) to interact with an environment to achieve a goal. The agent learns by receiving feedback in the form of rewards or penalties based on its actions. The goal is to learn a policy that maximizes cumulative rewards over time. Reinforcement learning is used in tasks where an agent must make a sequence of decisions, such as game playing, robotics, and autonomous driving.

 

\subsubsection{Classification and regression problems}
\label{subsec:class_reg}
The backbone of my work is centered around classification and regression tasks, so I work with supervised learning problems. To get a sense of what kind of projects I deal with, I introduce some brief examples for easier understanding.

Let's see a classification problem first: acceptance at university. Students are applying to a university, that accepts them based solely on their test points and grades. One of the students got 9 out of 10 (9/10) on the test and 8/10 on the grades. This student did well and got accepted. Another student got 3/10 on the test and 4/10 on the grades, they got rejected. And now we have a third student who got 7/10 on the test and 6/10 on the grades and we are wondering if they get accepted or not. There's an intuitive way to decide: looking at previous data. This scenario is demonstrated on the \ref{fig:classif}.~figure. In this figure, the horizontal axis shows the test results, and the vertical axis shows the grades. The first student I mentioned is marked with a green check mark (accepted), the second with a red X (rejected), and the third with a yellow question mark (we don't know yet if they are accepted or not). The red dots represent students who didn't get accepted, the blue dots mark accepted students, this is the previous data. Looking at the figure, we can draw a line (a red arrow pointing at it on the \ref{fig:classif}. figure), that separates the students based on their acceptance. We say that this line separates the students well, because most of the dots above the line are blue, and most of the dots under the line are red. Plotting the student with 7/10 test and 6/10 grade we can conclude that they are very likely to have been accepted because the corresponding point on \ref{fig:classif}. figure (yellow question mark) is above the line. This is a binary classification task because we split students into two categories: accepted and rejected.

Tasks with more than two categories are multi-class classification problems. My research also falls into this area, because the model I will introduce later (\ref{subsection:finish_line}) splits patients into three classes of cancer. To sum up, classification predicts discrete categorical labels.

\begin{figure}[tbh]
  \centering
  \includegraphics[scale=0.7]{classif_example.png}
  \captionsetup{justification=centering} 
  \caption{Acceptance at the example university. Students marked with red didn't get accepted, while the blue ones did.}
  \label{fig:classif}
  \small Source: \url{https://www.youtube.com/watch?v=46PywnGa_cQ}, 2024.05.13.
\end{figure}
\FloatBarrier


On the other hand, regression predicts continuous numerical values.
As a very brief example let's assume we have several years of data on July temperatures in Morocco (e.g. expressed in °C). Based on this, we want to predict this year's July temperature in Morocco. This regression task aims to approach the continuous function of this weather factor.

Further information about classification and regression problems is provided in Bishop's book \cite{bishop_book} (see the ``Linear Models for Regression'' and ``Linear Models for Classification'' sections).


\subsubsection{Neural networks}

A short introduction to neural networks is needed to understand the motivation behind my work.\\\\
\textbf{Multilayer perceptron}

\noindent My work mainly focuses on neural networks, so I will briefly present the one I deal with: the multilayer perceptron.

A multilayer perceptron (MLP) is a type of artificial neural network (ANN) that consists of multiple layers of nodes, or neurons, arranged in a feedforward manner, as the \ref{fig:nn}. figure shows. The first layer of neurons is called the input layer, the last is called the output layer and the layers between them are the hidden layers. Both the inputs and outputs of a neuron are numerical values.

In an MLP, information flows from the input layer, through one or more hidden layers, to the output layer, this is called forward propagation. Each neuron in a layer is connected to every neuron in the subsequent layer, and each connection is associated with a weight. The MLP learns from data by adjusting these weights through a process called backpropagation, which involves computing gradients and updating weights to minimize a loss function. Loss functions measure the difference between predicted and actual values (also called ``labels''), guiding machine learning models to minimize prediction errors during training. To understand exactly how MLPs and backpropagation algorithms work, and to see different loss functions and their meanings, I recommend this book for the interested reader \cite{bishop_book}.

MLPs are highly flexible and can learn complex non-linear relationships in data. They are capable of automatically extracting features from raw input data and can capture intricate patterns in high-dimensional datasets. This makes them suitable for a wide range of tasks, from image and speech recognition to natural language processing and beyond.

\begin{figure}[tbh]
  \centering
  \includegraphics[scale=0.8]{nn.png}
  \captionsetup{justification=centering} 
  \caption{A simplified model of a multilayer perceptron.}
  \label{fig:nn}
  \small Source: \cite{nn_img}
\end{figure}
\FloatBarrier


\noindent\textbf{Underfit, overfit and prevention techniques}\\

\noindent In machine learning, the goal is to create a model that can generalize well to unseen data. However, sometimes models can perform poorly due to underfitting or overfitting.

Underfitting occurs when a model is too simple (e.g. there are too few neurons in it) to capture the underlying structure of the data. A model that underfits performs poorly (meaning: with low accuracy) on both the training data and unseen data.

Overfitting occurs when a model is too complex (e.g. it contains too many neurons) and memorizes the noise present in the training data. Usually, a sign of overfitting can be that the model performs very well on the training data but poorly on unseen data. It's undesirable because it memorizes the training data rather than learning the general patterns.\\

To prevent underfitting and overfitting we often split the dataset into three disjoint parts: a training set, a validation set, and a test set.
The training set is used to train the model, the validation set is used to assess model performance while training, and the test set is used to evaluate the final model, after training.

To understand the \ref{fig:over_underfit}. figure, I first need to clarify the concepts of the number of epochs, the training error, and the validation error.

An epoch refers to one complete pass of the entire training dataset through the neural network. During each epoch, the model undergoes forward propagation (the prediction process), loss calculation (evaluating the loss function based on the prediction - given by the model - and the corresponding label), and backpropagation. Multiple epochs are typically used to allow the model to learn from the data iteratively.

The training error (or training loss) measures how well the model fits the training data, thus a lower training error indicates that the model fits the training data well. Training loss is typically calculated using a loss function that compares the predicted outputs of the model with the actual target values (labels) in the training dataset. Validation error (or validation loss) measures how well the model generalizes to unseen data. It's calculated in the same way as training error, by comparing the predicted outputs of the model with the actual target values, but in a separate dataset: the validation dataset. A lower validation loss indicates that the model is generalizing well to unseen data.

Ideally, we want both training error and validation error to be low. If a model has a low training error but a high validation error, it indicates overfitting. If both training and validation errors are high, it indicates underfitting.

In the \ref{fig:over_underfit}. figure the horizontal axis shows the number of epochs and the vertical axis shows the value of the loss function evaluated on both the training and the validation dataset in each epoch. In this figure, this model is underfitting before the 100th epoch (because both training and validation losses are high), and overfitting afterwards. Thus, we have to stop training at the 100th epoch (exactly where the purple arrow shows in the \ref{fig:over_underfit}. figure). Intuitively, this is what the early stopping algorithm does: tracks the validation loss, and when it starts to increase, it stops the training iteration. This avoids overfitting, as well as unnecessary computation time.

The interested reader can get acquainted with many other overfit prevention techniques including ensemble learning, regularization, and cross-validation by studying Murphy's book \cite{murphy_book}.


\begin{figure}[tbh]
  \centering
  \includegraphics[scale=0.4]{over_underfit2.png}
  \captionsetup{justification=centering} 
  \caption{The concept of underfit, overfit, and early stopping.}
  \label{fig:over_underfit}
  \small Source: \url{https://www.youtube.com/watch?v=b5934VsV3SA}, 2024.05.14.
\end{figure}
\FloatBarrier

\noindent\textbf{Data pre-processing}\\

\noindent The final building block for summing up the theory of my project is data pre-processing.

In real-world applications, raw data often comes in various forms, may contain noise, missing values, or inconsistencies, and may not be in a format directly usable by machine learning algorithms. Data preprocessing aims to address these issues and prepare the data for further analysis and model building. By employing appropriate preprocessing techniques, we can enhance the performance and reliability of our machine learning systems. I use dimensionality reduction, normalization, and imputation (replacing missing values with a calculated estimate - e.g. with median values).

In datasets with a large number of features (features are the input variables that represent the characteristics of the data - in my main work, genes will be the features), dimensionality reduction techniques like Principal Component Analysis (PCA) can be applied to reduce the number of features while preserving important information. This helps in reducing computational complexity, mitigating the curse of dimensionality, and improving model performance.

Normalization (scaling the value of features to a similar range - e.g. scaling every feature value to the [0,1] interval) ensures that all features contribute equally to the model.


\subsection{Starting point, previous works on this project}
\label{sec:prev_works}
I was new to the neural network, deep learning methodology, and the gene expression topic too at the start of the semester. Thus, I had no prior related knowledge or work. I only know about one similar work in my department, from which I got help for loading the appropriate dataset. It's named ``loading\_gene\_expression\_data.ipynb'', in the GitHub repository \ref{github}.

\newpage
%==================================================================
\section{Own work on project}
\label{sec:work}

\subsection{The basics}
\label{sec:subsection_one}

As a newcomer to machine learning and the Python programming language, I had to start with introductory courses. I completed several courses on Kaggle \cite{kaggle}: Python, Pandas, Intro to Machine Learning, and Intro to Deep Learning. This way I got acquainted with the tools I used throughout the semester. Also, I explored development environments and tools like Jupyter Notebooks (files with .ipynb extension in the referenced GitHub repository \ref{github}), Google Colaboratory \cite{colab}, and Kaggle \cite{kaggle}.

Every course followed the same structure: there were multiple topics within a course, and each topic had a theoretical and a practical (programming) part. So I could always test my knowledge to see if I understood the subject well enough.


After completing the mentioned courses I submitted solutions to the Titanic and Housing Prices competitions on Kaggle. In the Titanic solution, I used a model called RandomForestClassifier, which is designed for binary classification (categorizing data into two classes). I achieved a 77.5\% accuracy on this classification task. In the Housing Prices competition, being a regression task, I used a model called RandomForestRegressor and achieved a 17860 score, measured with the Mean Absolute Error (MAE) loss function. Both of my results are considered entry-level results. For further information about the courses and competitions I reference the Kaggle website \cite{kaggle}.

My corresponding programming work (courses and competitions) can be found in the directory named ``basics'' on my GitHub repository \ref{github}.

Now let's see what I've learned in detail. I summarized the completed courses in the following tables.

% -------- PYTHON TABLE -----------
\begin{table}[htbp]
  \centering
  \caption{Python course topics and their description}
\begin{tabularx}{\textwidth}{|m{5.5cm}|X|}
    \hline
 \textbf{Topic} & \textbf{Description} \\
 \hline
 Syntax, variable assignment, numbers & An overview of Python syntax, variable assignment, and arithmetic operators. \\
 \hline
 Functions &  Calling functions and defining our own, and using Python's builtin documentation. \\
 \hline
 Booleans and conditionals &   Using booleans and Python's if-then statements for branching logic. \\
 \hline
 Lists &  List indexing, slicing, and mutating.\\
 \hline
 Loops and list comprehension &  For and while loops, and a special Python feature: list comprehensions.  \\
 \hline
 Strings and dictionaries &   Two fundamental Python data types: strings (immutable), dictionaries (mutable). \\
 \hline
 Working with external libraries &  Builtin libraries, importing external libraries, operator overloading.\\
 \hline
\end{tabularx}
\end{table}

% -------- PANDAS TABLE -----------
\begin{table}[htbp]
  \centering
  \caption{Pandas course topics and their description}
\begin{tabularx}{\textwidth}{|m{5.5cm}|X|}
    \hline
 \textbf{Topic} & \textbf{Description} \\
 \hline
 Creating, reading, and writing & The two core objects: the DataFrame and the Series. Reading, writing, and inspecting CSV files. \\
 \hline
 Indexing, selecting, assigning &  Index-based, label-based, and conditional selection, assigning data. \\
 \hline
 Summary functions and maps &   Getting an overview of the dataset, and transforming data. \\
 \hline
 Grouping and sorting &  Groupwise analysis, multi-indexes, sorting.\\
 \hline
 Data types and missing values & Checking the data types, replacing missing values. \\
 \hline
 Renaming and combining &  Changing index names and/or column names, combining data from multiple DataFrames and/or Series.\\
 \hline
\end{tabularx}
\end{table}

% -------- Intro to ML TABLE -----------
\begin{table}[htbp]
  \centering
  \caption{Intro to Machine Learning course topics and their description}
\begin{tabularx}{\textwidth}{|m{5.5cm}|X|}
    \hline
 \textbf{Topic} & \textbf{Description} \\
 \hline
 How models work & Decision tree: the basic building block for some of the best models in data science.   \\
 \hline
 Basic data exploration &  Loading and interpreting the data using Pandas. \\
 \hline
 My first machine learning model &   Choosing features and building a model with scikit-learn library. \\
 \hline
 Model validation &  Measure the performance of the model, to test and compare alternatives. Metrics for model quality.\\
 \hline
 Underfitting and overfitting & Two of the most common model ``diseases''. \\
 \hline
 Random forests &  A forest of decision trees.\\
 \hline
 Machine learning competitions &  Getting to know the process of Kaggle competitions.\\
 \hline
\end{tabularx}
\end{table}
\FloatBarrier

% -------- Intro to DL TABLE -----------
\begin{table}[htbp]
  \centering
  \caption{Intro to Deep Learning course topics and their description}
\begin{tabularx}{\textwidth}{|m{5.5cm}|X|}
    \hline
 \textbf{Topic} & \textbf{Description} \\
 \hline
 A single neuron & Linear units, the building blocks of deep learning. Intro to Keras - a deep learning framework.  \\
 \hline
 Deep neural networks &  Hidden layers, activation functions, modularity. \\
 \hline
 Stochastic gradient descent &  The meaning of the loss function, the optimizer, the learning rate, and the batch size. \\
 \hline
 Overfitting and underfitting &  Learning curves, the model's capacity, and early stopping.\\
 \hline
 Dropout and batch normalization & Further techniques preventing overfit. \\
 \hline
 Binary classification &  Accuracy, cross-entropy, and the sigmoid function.\\
 \hline
\end{tabularx}
\end{table}


\subsection{Diving deeper into neural networks}
\label{subsection:diving_deeper}

After familiarizing myself with the basics, I had to delve deeper into the topic of neural networks to build more robust and powerful models. I've read chapters 1-4. from the book of Gábor Horváth et al. \cite{horvath_book} about neural networks, and I also completed an introductory course into deep learning, using PyTorch as a framework \cite{udacity}. Let's see the results of my programming work - implementing the todo-s in the notebooks - in this course. Every directory and file I reference in this subsection can be found in my GitHub repository under the ``udacity\_work'' directory \ref{github}. The datasets I mention are well-known, benchmark datasets and can be found on Kaggle \cite{kaggle}.

First, I implemented some of the steps in the training of a neural network that analyses student data. It included one-hot encoding, scaling, and some of the backpropagation steps. My solution is in the StudentAdmissionSulutions.ipynb.

Then I moved on to getting to know PyTorch. I made programming work with tensors, training neural networks, the Fashion-MNIST dataset, inference and validation, loading image data, and transfer learning. These projects can be found in the ``pytorch\_intro'' direcory.

After I felt comfortable with PyTorch I learned about Convolutional Neural Networks (CNNs) and did projects like building an MLP for the MNIST dataset, implementing filters, visualizing layers (convolutional and max pooling), and designing a CNN for classifying the CIFAR-10 dataset. The \ref{fig:cifar10_cnn}. figure shows some of images classified by the CNN. Over each picture, there are written two categories. The model's prediction is the one before the brackets and the actual category is between the brackets. If the two are the same, it's written in green, else in red. It was a randomly chosen sample, so we can conclude, that this model does a pretty good job of classifying the images because there are only 2 red labels over the displayed 16 images. It reached 76\%, but everyone can run the Jupiter Notebook (it is named: cifar10\_cnn\_solution.ipynb) and see the test result locally, it will be around 70\% every time. My corresponding work is in the ``cnn'' directory. 

\begin{figure}[tbh]
  \centering
  \includegraphics[scale=0.5]{cifar10_cnn.png}
  \captionsetup{justification=centering} 
  \caption{A part of the classified CIFAR-10 images with CNN.}
  \label{fig:cifar10_cnn}
\end{figure}
\FloatBarrier

An interesting application of CNNs is the style transfer task. In the Style\_transfer\_Solution.ipynb notebook I recreated a style transfer method that is outlined in the paper by Gatys et al. \cite{style_transfer}. An example is shown in the \ref{fig:style_transfer}. figure, where the content image is of a woman, and the style image is one of Robert Delaunay's paintings. The generated target image still contains the woman but is stylized with the circles and rainbow colors. The fundamental idea behind this problem is to separate the content and the style of an image. It can be done with mathematical methods (e.g. using the Gram matrix), which is explained in \cite{style_transfer}.

\begin{figure}[tbh]
  \centering
  \includegraphics[scale=0.25]{style_transfer.jpg}
  \captionsetup{justification=centering} 
  \caption{The result of style transfer.}
  \label{fig:style_transfer}
\end{figure}
\FloatBarrier

There was only one big chapter left in this course, the topic of Recurrent Neural Networks (RNNs). Here I learned about Long Short Term Memory (LSTM) models, the gates that these architectures are using (namely: the learn, forget, remember, and the use gate), then I implemented an LSTM that makes character-level text prediction (Character\_Level\_RNN\_Solution.ipynb). It takes a text as input and predicts the following N (number of characters to predict, given by the user) characters of that text. Another remarkable application idea is to make a sentiment prediction LSTM. In the Sentiment\_RNN\_Solution.ipynb I built a model that takes a review as input and predicts whether the review is positive or negative.

At the end of this learning journey, I got an insight into deploying ML models. The ``Deploying PyTorch models'' topic of the course explains how we can translate Python code into C++ with PyScript, achieving a much faster execution time.

To close off this subsection, I must mention my MLP solutions for the Titanic and Housing Prices Kaggle competitions. On the Titanic, I achieved a 1.2\% better accuracy than with the RandomForest solution in the previous subsection. In the Housing Prices competition, I reached a much worse MAE score, than before. Both projects combined my knowledge learned from the courses, including data pre-processing, normalization, designing model architecture, training, validating, and testing the model. These works can be found in the ``mlp\_kaggle'' directory.


\subsection{The finish line}
\label{subsection:finish_line}

After learning all the necessary theory and trying out my abilities to build simple MLPs, my main task has come: classifying cancer patients based on their gene expression data. First, I got to know the Sztaki server environment: using a Linux terminal, starting a Docker container, and doing my work within this. I watched a video about gene expression \cite{gene_expression}, to understand the cBioPortal dataset \cite{cbio}. The programming works and the data is in the ``main\_task'' directory.

First, I loaded three mRNA expression tables. One contains information about breast cancer, another one about glioblastoma (a type of brain tumor), and the last one about ovarian cancer patients. Every table contains many missing values, unscaled data, and more than 10000 genes. Thus, I applied imputation, replacing missing values with the corresponding median value. Then I applied dimensionality reduction (PCA) because I chose genes as my features and there were more than 10000 of them. This way, important information was kept in the data, but I got 200 features instead of the original 10000+, so I reduced computation time as well. I used normalization too, in the form of min-max scaling, so after PCA every value was mapped into the [0,1] interval. Consequently, input values don't vary on a too large scale, so they're far better for training than unnormalized values. The reason behind this is the fact that deep learning models use gradient calculation and large scale data makes the training process more unstable and slower. The exact math can be found in Gábor Horváth's book \cite{horvath_book}.

Data was cleaned, so I joined the different cancer types' DataFrame into one dataset, I shuffled the samples (to reduce overfitting chances) then I divided it into three parts: train, validation, and test dataset. I had to assign the tables with labels too, so I created the following three categories: Breast cancer, Glioblastoma, and Ovarian cancer. In magnitude, it was a 2800x200 array, meaning about 2800 patients' data and the compressed 200 gene features.

After I pre-processed the data I used a logistic regression model first to predict the probabilities of the categories. Then it classified the patients into the three classes I mentioned, based on the highest probability calculated in the previous step. It achieved 92\% accuracy on the test set. It got all the breast cancer and ovarian cancer samples right but found none of the glioblastoma patients. My conclusion was that it had too few samples and the model couldn't learn the patterns behind it, an underfit phenomenon occured. So I loaded a twice as big dataset for the glioblastoma samples, and with it, the model reached 100\% accuracy on the test set.

For interest, I built an MLP for this task too. I say for interest, because the flawless logistic regression solution is simpler, so it perfectly does the job. But when designing the MLP, I kept in mind that I had a relatively small dataset (about 2800x200). So a few layers and nodes may be enough, to avoid overfitting, thus I chose only one hidden layer with 32 neurons. The net has 3 output neurons because there are 3 classes right now (breast cancer, glioblastoma, ovarian cancer). This is one of the future project ideas, to make more classes, and this way to enhance the the width of the model. The other one is to train on more data samples from each cancer type, thus deepening the model's abilities.

\subsection{Summary}
\label{sec:summary}

My main task was to build a model that could classify cancer patients with reasonable accuracy (above 70\%). As stated in the introduction \ref{sec:introduction}, AI has powerful applications in genomics, thus my task is a prospective one to enhance machine learning models' capabilities in this area. My solution supports this statement, as I have built a flawless classifier for three cancer types.

I used all the techniques listed in the introduction, but most importantly: data pre-processing, multilayer perceptrons, and logistic regression models. In the topic of data pre-processing my most used tools were normalization, imputation and dimensionality reduction (PCA).

The most notable result is the 100\% accurate classifier, trained on real cancer data, collected by US hospitals \cite{cbio}. Also, remember that I spotted an underfit phenomenon and resolved the issue by loading more data. The achieved accuracies in the mentioned Kaggle competitions (Titanic and Housing Prices) are also worthwhile because they mark which directions for development are prospective and which aren't.

In conclusion, this field has much more potential than described in this paper, so I will continue my research enhancing this model's width (increasing the number of investigated cancer types) and depth (training on more samples in each cancer type).

\newpage
%==================================================================
% --------- REFERENCES --------
\begin{thebibliography}{9}

\bibitem{aibook} Russell, S., \& Norvig, P. (Pearson, 2009). Artificial Intelligence: A Modern Approach. 3rd Edition.

\bibitem{bishop_book}
    Bishop, C. M. (Springer, 2006). Pattern Recognition and Machine Learning.
    
\bibitem{dlbook} Goodfellow, I., Bengio, Y., \& Courville, A. (MIT Press, 2016).
    Deep Learning.
    \url{http://www.deeplearningbook.org}, 2024.05.12.

\bibitem{murphy_book} Murphy, K. (MIT Press, 2012) Machine Learning. A Probabilistic Perspective.

\bibitem{horvath_book} Horváth, G. et al. (Panem Könyvkiadó Kft., 2006) Neurális hálózatok.

\bibitem{dl} LeCun, Y., Bengio, Y., \& Hinton, G. (2015). Deep learning. \emph{Nature, 521(7553), 436–444}

\bibitem{dl_bio} Alipanahi, B., Delong, A., Weirauch, M. et al. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. \emph{Nat Biotechnol 33, 831–838.}

\bibitem{dl_gene} Angermueller C, Pärnamaa T, Parts L, Stegle O. (2016). Deep learning for computational biology. \emph{Mol Syst Biol.}

\bibitem{nn_img} Miguel-A-Ramirez et al. (2022). Poisoning Attacks and Defenses on Artificial Intelligence: A Survey.

\bibitem{style_transfer} Leon A. Gatys et al. (2016). Image Style Transfer Using Convolutional Neural Networks.\ \emph{2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR).}


\bibitem{cbio} \emph{Dataset on cancer patients, collected by US hospitals.}
  \url{https://www.cbioportal.org/datasets}, 2024.05.12.

\bibitem{kaggle} \emph{Kaggle - a data science community.}
  \url{https://www.kaggle.com}, 2024.05.14.

\bibitem{udacity} \emph{Intro to Deep Learning with Pytorch.}
    \url{https://www.udacity.com/course/deep-learning-pytorch--ud188}, 2024.05.14.

\bibitem{gene_expression} \emph{Gene expression video.}
    \url{https://www.youtube.com/watch?v=X4oxrewkDpQ}, 2024.05.14.
    
\bibitem{colab} \emph{Google Colaboratory}
    \url{https://colab.research.google.com}, 2024.05.15.
  

\end{thebibliography}


\subsection*{Attached documents}
\label{sec:attached_documents}
\begin{itemize}
    \item The GitHub repository containing my programming work: \url{https://github.com/flash4242/project_lab}
    \label{github}
\end{itemize}

\end{document} 

%%% Local Variables: 
%%% mode: latex 
%%% TeX-master: t 
%%% End:

