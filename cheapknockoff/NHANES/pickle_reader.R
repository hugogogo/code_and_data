#### This is the source code for reading the diabetes data from the raw NHANES dataset
rm(list = ls())
library(reticulate)
file_name <- "./diabetes_all.pkl"

py_install("pandas")
source_python("pickle_reader.py")

raw <- read_pickle_file(file_name)

raw$costs <- as.integer(raw$costs)

# hand coded feature names/descriptions/types
# using implementation details from https://github.com/mkachuee/Opportunistic/blob/master/nhanes.py
raw$type <- c(rep("Demographics (real)", 2),
              rep("Demographics (one hot)", 11),
              rep("Demographics (real)", 2),
              rep("Examination (real)", 13),
              rep("Laboratory (real)", 4),
              rep("Questionnaire (real)", 6),
              rep("Questionnaire (one hot)", 4),
              rep("Questionnaire (real)", 3)
)

raw$var_name <- c("RIAGENDR", 
                  rep("RIDAGEYR", 7),
                  rep("RIDRETH3", 4),
                  "RIDRETH1",
                  "INDHHINC",
                  "DMDEDUC2",
                  "BPXSY1",
                  "BPXDI1",
                  "BPXSY2",
                  "BPXDI2",
                  "BPXSY3",
                  "BPXDI3",
                  "BPXSY4",
                  "BPXDI4",
                  "BMXBMI",
                  "BMXWAIST",
                  "BMXHT",
                  "BMXLEG",
                  "BMXWT",
                  "LBXTC",
                  "LBXTR",
                  "LBXFB",
                  "LBDLDL",
                  "ALQ101",
                  "ALQ120Q",
                  "PAQ605",
                  "PAQ620",
                  "PAQ180",
                  "SLD010H",
                  rep("SMQ020", 4),
                  "SMD030",
                  "MCQ250A",
                  "BPQ020"
)

raw$name <- c("Gender",
              "Age",
              rep("Race_3", 7),
              rep("Race_1", 4),
              "Income",
              "Education",
              "Systolic BP(1st)",
              "Diastolic BP(1st)",
              "Systolic BP(2nd)",
              "Diastolic BP(2nd)",
              "Systolic BP(3rd)",
              "Diastolic BP(3rd)",
              "Systolic BP(4th)",
              "Diastolic BP(4th)",
              "Body mass index",
              "Waist circumference",
              "Height",
              "Upper leg length",
              "Weight",
              "Cholesterol",
              "Triglyceride",
              "Fibrinogen",
              "LDL-cholesterol",
              "Alcohol amount",
              "Alcohol frequency",
              "Vigorous activity",
              "Moderate activity",
              "Avg physical activity level",
              "Avg sleep hours",
              rep("Smoked at least 100 cigarettes in life", 4),
              "Age started smoking",
              "Relatives having diabetes",
              "High blood pressure history"
)

raw$label <- c("Gender of the sample person",
               "Best age in years of the sample person at time of HH screening. Individuals 85 and over are topcoded at 85 years of age",
               rep("Recode of reported race and Hispanic origin information, with Non-Hispanic Asian Category", 7),
               rep("Recode of reported race and ethnicity information", 4),
               "Total household income (reported as a range value in dollars)",
               "(SP Interview Version) What is the highest grade or level of school {you have/SP has} completed or the highest degree {you have/s/he has} received?",
               "Systolic: Blood pressure (1st reading) mm Hg",
               "Diastolic: Blood pressure (1st reading) mm Hg",
               "Systolic: Blood pressure (2nd reading) mm Hg",
               "Diastolic: Blood pressure (2nd reading) mm Hg",
               "Systolic: Blood pressure (3rd reading) mm Hg",
               "Diastolic: Blood pressure (3rd reading) mm Hg",
               "Systolic: Blood pressure (4th reading) mm Hg",
               "Diastolic: Blood pressure (4th reading) mm Hg",
               "Body Mass Index (kg/m**2)",
               "Waist Circumference (cm)",
               "Standing Height (cm)",
               "Upper Leg Length (cm)",
               "Weight (kg)",
               "Total cholesterol (mg/dL)",
               "Triglyceride (mg/dL)",
               "Fibrinogen (mg/dL)",
               "LDL-cholesterol (mg/dL)",
               "The next questions are about drinking alcoholic beverages. Included are liquor (such as whiskey or gin), beer, wine, wine coolers, and any other type of alcoholic beverage. In any one year, {have you/has SP} had at least 12 drinks of any type of alcoholic beverage? By a drink, I mean a 12 oz. beer, a 4 oz. glass of wine, or an ounce of liquor",
               "In the past 12 months, how often did {you/SP} drink any type of alcoholic beverage? PROBE: How many days per week, per month, or per year did {you/SP} drink?",
               "Next I am going to ask you about the time {you spend/SP spends} doing different types of physical activity in a typical week. Please answer these questions even if {you do not consider yourself/SP does not consider himself/herself} to be a physically active person. Think first about the time {you spend/SP spends} doing work. Think of work as the things that {you have/SP has} to do such as paid or unpaid work, studying or training, household chores, and yard work. In answering the following questions, 'vigorous-intensity activities' are activities that require hard physical effort and cause large increases in breathing or heart rate, and 'moderate-intensity activities' are activities that require moderate physical effort and cause small increases in breathing or heart rate. Does {your/SP's} work involve vigorous-intensity activity that causes large increases in breathing or heart rate like carrying or lifting heavy loads, digging or construction work for at least 10 minutes continuously?",
               "Does {your/SP's} work involve moderate-intensity activity that causes small increases in breathing or heart rate such as brisk walking or carrying light loads for at least 10 minutes continuously?",
               "Please tell me which of these four sentences best describes {your/SP's} usual daily activities? [Daily activities may include {your/his/her} work, housework if {you are/s/he is} a homemaker, going to and attending classes if {you are/s/he is} a student, and what {you/s/he} normally {do/does} throughout a typical day if {you are/he/she is} a retiree or unemployed.]",
               "The next set of questions is about your sleeping habits. How much sleep {do you/does SP} usually get at night on weekdays or workdays?",
               rep("These next questions are about cigarette smoking and other tobacco use. {Have you/Has SP} smoked at least 100 cigarettes in {your/his/her} entire life?", 4),
               "How old {were you/was SP} when {you/s/he} first started to smoke cigarettes fairly regularly?",
               "Including living and deceased, were any of {SP's/ your} biological that is, blood relatives including grandparents, parents, brothers, sisters ever told by a health professional that they had diabetes?",
               "{Have you/Has SP} ever been told by a doctor or other health professional that {you/s/he} had hypertension, also called high blood pressure?"
)

raw$one_hot_idx <- c(3:13, 39:42)

save(raw, file = "diabetes.RData")
