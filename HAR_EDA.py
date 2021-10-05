# import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# features names are not present in dataset so lets read features from features.txt
features = list()
with open('UCI_HAR_Dataset/features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]

print("Number of features: ", len(features))
# Number of features:  561

# Obtain train dataset

X_train = pd.read_csv('UCI_HAR_Dataset/train/X_train.txt', delim_whitespace=True, header=None, encoding='latin-1')

# add column names to dataframe
X_train.columns = features

# add subject column to dataframe
X_train['subject'] = pd.read_csv('UCI_HAR_dataset/train/subject_train.txt', header=None, squeeze=True)

y_train = pd.read_csv('UCI_HAR_dataset/train/y_train.txt', names=['Activity'], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})

# add other features to single dataframe
train = X_train
train['Activity'] = y_train
train['ActivityName'] = y_train_labels

train.sample()
""" tBodyAcc-mean()-X  tBodyAcc-mean()-Y  tBodyAcc-mean()-Z  ...  subject  Activity  ActivityName
1832           0.261125          -0.022179          -0.106831  ...        8         6        LAYING """

train.shape
# (7352, 564)

# Obtain test dataset
X_test = pd.read_csv('UCI_HAR_dataset/test/X_test.txt', delim_whitespace=True, header=None)
X_test.columns = features

X_test['subject'] = pd.read_csv('UCI_HAR_dataset/test/subject_test.txt', header=None, squeeze=True)

y_test = pd.read_csv('UCI_HAR_dataset/test/y_test.txt', names=['Activity'], squeeze=True)
y_test_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING'})

test = X_test
test['Activity'] = y_test
test['ActivityName'] = y_test_labels

test.sample()
""" tBodyAcc-mean()-X  tBodyAcc-mean()-Y  tBodyAcc-mean()-Z  ...  subject  Activity        ActivityName
754           0.241562          -0.050591          -0.113984  ...        9         3  WALKING_DOWNSTAIRS """

test.shape
# (2947, 564)

# DATA CLEANING
# 0.1: Check for duplicates

print('Duplicate rows in train data: ', sum(train.duplicated()))
print('Duplicate rows in test data: ', sum(test.duplicated()))
# Duplicate rows in train data:  0
# Duplicate rows in test data:  0

# 0.2: Check for null values

print('Null values in train data: ',train.isnull().values.sum())
print('Null values in test data: ',test.isnull().values.sum())
# Null values in train data:  0
# Null values in test data:  0

# 0.3: Check for dataset imbalance
sns.set_style('whitegrid')

plt.figure(figsize=(16,8))
plt.title('Data provided by each user', fontsize=20)
sns.countplot(x='subject',hue='ActivityName', data = train)
plt.savefig('0.3_check_data_imbalance.png')
plt.show()

# observation: almost same number of readings from all subject

plt.title('No of Datapoints per Activity', fontsize=15)
sns.countplot(train.ActivityName)
plt.xticks(rotation=90)
plt.savefig('0.3_check_data_imbalance2.png')
plt.show()

# observation: data is almost well balanced

# 0.4: rename features names to appropriate names

columns = train.columns
columns = train.columns

# Remove '()' from column names
columns = columns.str.replace('[()]','')

# Remove '-' from column names
columns = columns.str.replace('[-]', '')

# Remove ',' from column names
columns = columns.str.replace('[,]','')

# update column names
train.columns = columns
test.columns = columns

test.columns
""" Index(['tBodyAccmeanX', 'tBodyAccmeanY', 'tBodyAccmeanZ', 'tBodyAccstdX',
       'tBodyAccstdY', 'tBodyAccstdZ', 'tBodyAccmadX', 'tBodyAccmadY',
       'tBodyAccmadZ', 'tBodyAccmaxX',
       ...
       'angletBodyAccMeangravity', 'angletBodyAccJerkMeangravityMean',
       'angletBodyGyroMeangravityMean', 'angletBodyGyroJerkMeangravityMean',
       'angleXgravityMean', 'angleYgravityMean', 'angleZgravityMean',
       'subject', 'Activity', 'ActivityName'],
      dtype='object', length=564) """

# 0.5 save dataframe into csv file
train.to_csv('UCI_HAR_Dataset/csv_files/train.csv', index=False)
test.to_csv('UCI_HAR_Dataset/csv_files/test.csv', index=False)

# EDA

# 1.0: Differentiate between stationary and moving activities
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(train, hue='ActivityName', size=6,aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMagmean', hist=False)\
    .add_legend()
plt.annotate("Stationary Activities", xy=(-0.956,17), xytext=(-0.9, 23), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.savefig('1.0_stationary_vs_moving_activities_diff.png')
plt.show()

# for plotting purposes taking datapoints of each activity to a different dataframe
df1 = train[train['Activity']==1]
df2 = train[train['Activity']==2]
df3 = train[train['Activity']==3]
df4 = train[train['Activity']==4]
df5 = train[train['Activity']==5]
df6 = train[train['Activity']==6]

plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
plt.title('Stationary Activities(Zoomed in)')
sns.distplot(df4['tBodyAccMagmean'],color = 'r',hist = False, label = 'Sitting')
sns.distplot(df5['tBodyAccMagmean'],color = 'm',hist = False,label = 'Standing')
sns.distplot(df6['tBodyAccMagmean'],color = 'c',hist = False, label = 'Laying')
plt.axis([-1.01, -0.5, 0, 35])
plt.legend(loc='center')

plt.subplot(2,2,2)
plt.title('Moving Activities')
sns.distplot(df1['tBodyAccMagmean'],color = 'red',hist = False, label = 'Walking')
sns.distplot(df2['tBodyAccMagmean'],color = 'blue',hist = False,label = 'Walking Up')
sns.distplot(df3['tBodyAccMagmean'],color = 'green',hist = False, label = 'Walking down')
plt.legend(loc='center right')

plt.tight_layout()
plt.savefig('1.0_stationary_vs_moving_activities_diff2.png')
plt.show()

# 1.1: magnitude of acceleration can separate well
plt.figure(figsize=(7,7))
sns.boxplot(x='ActivityName', y='tBodyAccMagmean',data=train, showfliers=False, saturation=1)
plt.ylabel('Acceleration Magnitude mean')
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=90)
plt.savefig('1.1_Acceleration Magnitude mean.png')
plt.show()

""" Observations:

If tAccMean is < -0.8 then the Activities are either Standing or Sitting or Laying.
If tAccMean is > -0.6 then the Activities are either Walking or WalkingDownstairs or WalkingUpstairs.
If tAccMean > 0.0 then the Activity is WalkingDownstairs.
We can classify 75% the Acitivity labels with some errors. """

# 1.2: Position of GravityAccelerationComponants differentiates well
sns.boxplot(x='ActivityName', y='angleXgravityMean', data=train)
plt.axhline(y=0.08, xmin=0.1, xmax=0.9,c='m',dashes=(5,3))
plt.title('Angle between X-axis and Gravity_mean', fontsize=15)
plt.xticks(rotation = 40)
plt.savefig('1.2_Angle between X-axis and Gravity_mean.png')
plt.show()

""" 
Observations:

If angleX,gravityMean > 0 then Activity is Laying.
We can classify all datapoints belonging to Laying activity with just a single if else statement. """

sns.boxplot(x='ActivityName', y='angleYgravityMean', data = train, showfliers=False)
plt.title('Angle between Y-axis and Gravity_mean', fontsize=15)
plt.xticks(rotation = 40)
plt.axhline(y=-0.22, xmin=0.1, xmax=0.8, dashes=(5,3), c='m')
plt.savefig('1.2_Angle between Y-axis and Gravity_mean.png')
plt.show()