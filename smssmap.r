#Filtering mobile phone spam with Naive Bayes Algorithm
#Spam and Ham 

sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

str(sms_raw)

sms_raw$type <- factor(sms_raw$type)

str(sms_raw$type)

table(sms_raw$type)

install.packages("tm")


library(tm)

sms_corpus <- VCorpus(VectorSource(sms_raw$text))

print(sms_corpus)

inspect(sms_corpus)

#to view actual message text 

as.character(sms_corpus[[1]])

lapply(sms_corpus[1:2], as.character)

#the th_map() function provides a method to apply  a transformation (also known as mapping)

#converting captial to lower case
sms_corpus_clean <- tm_map(sms_corpus,
                           content_transformer(tolower))

as.character(sms_corpus_clean[[1]])

#removing numbers 
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

#removing stop words 
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

#removing punctuation 
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)


#removePunctuation("hello........world")


install.packages("SnowballC")

library(SnowballC)

wordStem(c("learn", "learned", "learning", "learns"))

#wordstem function for entire document is = stemDocument
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)


#After removing numbers, stop words and punctuations as well as performing stemming, the text messages are left with the blank spaces 

#Remove additional white spaces (stripWhitespace())
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

#After these cleaning of data 

as.character(sms_corpus_clean[[1]])

#The final step is to spilt the messages into individual components through a process called as Tokenization

#Using DocumentTermMatrix funtion 

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)



#Creating Training and Test datasets


sms_dtm_train <- sms_dtm[1:4172, ]
sms_dtm_test <- sms_dtm[4173:5572, ]

#Pulling labels from raw data

sms_train_labels <- sms_raw[1:4172, ]$type
sms_test_labels <- sms_raw[4173:5572, ]$type



#Comparing proportions of ham and spam in traing and test dataframes

prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))


#WordCloud is a way to visually depict the frequency at which words appear  in text data
install.packages("wordcloud")

library(wordcloud)

wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

#wordcloud(sms_corpus_clean, min.freq = 80, random.order = FALSE,scale=c(3.5,0.25))


#Creating word Cloud for both spam and ham

spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")


wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

#Finding frequency of the words
findFreqTerms(sms_dtm_train, 5)

sms_freq_words <- findFreqTerms(sms_dtm_train, 5)


str(sms_freq_words)

#We now need to filter out DTM to include only the terms appearing in a specified vector

sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]


convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}



sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test , MARGIN = 2, convert_counts)


#TRAINING a Model on the Data

install.packages("e1071")
library(e1071)


sms_classifier <- naiveBayes(sms_train, sms_train_labels)


sms_test_pred <- predict(sms_classifier, sms_test)

install.packages("gmodels")

library(gmodels)

CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE,
           prop.t = FALSE,
           dnn = c('predicted','actual'))



#Improving the model performance
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)


sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE,
           prop.t = FALSE,
           prop.r = FALSE,
           dnn = c('predicted','actual'))



