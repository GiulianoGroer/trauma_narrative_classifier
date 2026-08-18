#--------------------------------------------------------------------------------------------------------------------#
####---loading packages---####
#--------------------------------------------------------------------------------------------------------------------#

library(dplyr)
library(lmtest)
library(sandwich)
library(psych)
library(tidyr)
library(gvlma)
library(logistf)
library(glmnet)
library(pROC)
library(ggplot2)
#--------------------------------------------------------------------------------------------------------------------#
####---data loading---####
#--------------------------------------------------------------------------------------------------------------------#
#wd and data
setwd("C:/Users/Giuliano.DESKTOP-NPATJ24/Desktop/PTSD_online_survey/data")
dataset <- read.csv2("dataset_preprocessed.csv", sep=";", header=TRUE, encoding='utf-8')

#make columns numerics
cols <- grep("_order_|_wc|_sc", names(dataset)) # indices of numeric columns to treat as numeric
dataset[cols] <- lapply(dataset[cols], as.numeric)

#calculate average sentence length
dataset$sentence_length_trauma <- dataset$response_trauma_wc / dataset$response_trauma_sc
dataset$sentence_length_neutr <- dataset$response_neutr_wc / dataset$response_neutr_sc
dataset$sentence_length_negnt <- dataset$response_negnt_wc / dataset$response_negnt_sc

#Convert "condition_pcl" to numeric: subclinical -> 0, clin_pcl -> 1
dataset$condition_pcl_numeric <- ifelse(dataset$condition_pcl == "clin_pcl", 1,
                                            +ifelse(dataset$condition_pcl == "subclinical", 0, NA))
dataset <- dataset%>%
  mutate(education = ifelse(education == 1,"Kein Schulabschluss",
                            ifelse(education == 2, "Erster allgemeinbildender Schulabschluss",
                                   ifelse(education == 3, "Mittlerer Schulabschluss/Fachoberschulreife",
                                          ifelse(education == 4, "Fachhochschulreife",
                                                 ifelse(education == 5, "Allgemeine Hochschulreife",
                                                        ifelse(education == 6, "Bachelor",
                                                               ifelse(education == 7, "Master",
                                                                      ifelse(education == 8, "Promotion","")))))))))
dataset <- dataset%>%
  mutate(socioeconomic = ifelse(socioeconomic == 4,"0-9999€",
                                ifelse(socioeconomic == 5, "10.000-24.999€",
                                       ifelse(socioeconomic == 6, "25.000-49.999€",
                                              ifelse(socioeconomic == 7, "50.000-74.999€",
                                                     ifelse(socioeconomic == 8, "75.000-99.999€",
                                                            ifelse(socioeconomic == 9, "100.000-149.999€",
                                                                   ifelse(socioeconomic == 10, "150.000 and more",
                                                                          ifelse(socioeconomic == 11, "Did not answer","")))))))))
ptsd <- subset(dataset,dataset$condition_pcl_numeric ==1)
non_ptsd <- subset(dataset,dataset$condition_pcl_numeric ==0)

dass_d <- subset(dataset, dataset$condition_dass_depression == "clin_dass_depression")
non_dass_d<- subset(dataset, dataset$condition_dass_depression == "subclinical_dass_depression")

dass_a <- subset(dataset, dataset$condition_dass_anxiety == "clin_dass_anxiety")
non_dass_a<- subset(dataset, dataset$condition_dass_anxiety == "subclinical_dass_anxiety")

dass_s <- subset(dataset, dataset$condition_dass_stress == "clin_dass_stress")
non_dass_s<- subset(dataset, dataset$condition_dass_stress == "subclinical_dass_stress")

#--------------------------------------------------------------------------------------------------------------------#
####--- Demographic data ---####
#--------------------------------------------------------------------------------------------------------------------#
#Age
describe(dataset$age)
hist(dataset$age)
describe(ptsd$age)
describe(non_ptsd$age)

#Gender
table(dataset$gender)
table(ptsd$gender)
table(non_ptsd$gender)

#Income
table(dataset$socioeconomic)
table(ptsd$socioeconomic)
table(non_ptsd$socioeconomic)

#Education

table(dataset$education)
table(ptsd$education)
table(non_ptsd$education)


dataset %>%
  mutate(told_event_lec5 = factor(told_event_lec5,levels = 1:17,
                                  labels = c("Natural disaster","Fire or explosion","Transportation accident","Serious accident at work/home",
                                             "Exposure to toxic substance","Physical assault","Assault with weapon",
                                             "Sexual assault","Other unwanted sexual experience","Combat or war exposure","Captivity","Life-threatening illness/injury",
                                             "Severe human suffering","Sudden violent death","Sudden accidental death","Serious injury or death caused to someone else", "something else"))) %>%
  count(gender, told_event_lec5, .drop = FALSE) %>%
  group_by(gender) %>%
  mutate(percent = n / sum(n) * 100) %>%
  ungroup()%>%
  pivot_wider(
    names_from = gender,
    values_from = c(n, percent),
    values_fill = 0)


ggplot(dataset, aes(x = response_trauma_wc)) +
  geom_histogram(
    binwidth = 5,
    boundary = 0.5,
    fill = "#4C78A8",
    color = "white",
    linewidth = 0.3
  ) +
  labs(
    x = "Trauma response score",
    y = "Frequency"
  ) +
  scale_x_continuous(
    breaks = scales::pretty_breaks()
  ) +
  scale_y_continuous(breaks = scales::pretty_breaks())+
  theme_classic(base_size = 12)

describe(dataset$response_trauma_wc)


#--------------------------------------------------------------------------------------------------------------------#
####--- distribution tests ---####
#--------------------------------------------------------------------------------------------------------------------#

#distribution of trauma narrative LEC-5 categories across PTSD-risk groups

chisq.test(table(dataset$condition_pcl_numeric, dataset$told_event_lec5))

chisq.test(table(dataset$condition_pcl_numeric, dataset$gender))
t.test(subset(dataset$age ,dataset$condition_pcl_numeric == 0), subset(dataset$age, dataset$condition_pcl_numeric == 1))

#--dass_d--#
chisq.test(table(dataset$condition_dass_depression, dataset$told_event_lec5))
chisq.test(table(dataset$condition_dass_depression, dataset$gender))
t.test(dass_d$age, non_dass_d$age)

#--dass_s--#
chisq.test(table(dataset$condition_dass_stress, dataset$told_event_lec5))
chisq.test(table(dataset$condition_dass_stress, dataset$gender))
t.test(dass_s$age, non_dass_s$age)

#--dass_a--#
chisq.test(table(dataset$condition_dass_anxiety, dataset$told_event_lec5))
chisq.test(table(dataset$condition_dass_anxiety, dataset$gender))
t.test(dass_a$age, non_dass_a$age)

hist(dataset$DASS21_anxiety)
table(dataset$DASS21_depression)


